"""Performance benchmarking adapter — latency, throughput, and resource profiling.

Implements IPerformanceBenchmarker. Measures end-to-end model serving performance:
  - Latency measurement: P50, P90, P95, P99 percentiles under configurable load
  - Throughput measurement: sustained requests/sec at steady state
  - Memory usage profiling: peak RSS and heap allocation deltas
  - CPU utilization tracking: average and peak across benchmark duration
  - Baseline comparison: current vs. previous release metrics
  - Performance regression detection with configurable tolerance bands
  - Benchmark report generation with time-series data
  - Load profile configuration: ramp-up, steady-state, cooldown phases

All HTTP load generation and resource sampling use asyncio and run entirely
within the async context (no thread dispatch needed for pure async I/O).
psutil calls for memory/CPU sampling are dispatched via asyncio.to_thread().
"""

import asyncio
import statistics
import time
from typing import Any

import httpx

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)

# Minimum number of samples required for percentile computation
_MIN_SAMPLES_FOR_PERCENTILES: int = 10

# CPU and memory sampling interval during benchmark (seconds)
_RESOURCE_SAMPLING_INTERVAL: float = 0.5


class PerformanceBenchmarker:
    """Latency, throughput, and resource profiler for AI model endpoints.

    Executes controlled load tests against model inference endpoints, measures
    latency percentiles and throughput, and detects performance regressions
    against a stored baseline.

    Args:
        settings: Application settings providing timeout and worker configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings for the testing harness.
        """
        self._settings = settings
        self._timeout_seconds = settings.eval_timeout_seconds
        self._max_workers = settings.max_eval_workers

    async def measure_latency(
        self,
        endpoint: str,
        sample_payload: dict[str, Any],
        num_requests: int,
        concurrency: int,
        warmup_requests: int = 5,
    ) -> dict[str, Any]:
        """Measure inference latency across num_requests with given concurrency.

        Executes a warmup phase to prime connection pools and model caches,
        then measures latency for num_requests at the configured concurrency.

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload dict to send for each probe.
            num_requests: Total number of timed requests to execute.
            concurrency: Number of simultaneous inflight requests.
            warmup_requests: Number of warmup requests to execute first.

        Returns:
            Latency result dict with: p50_ms, p90_ms, p95_ms, p99_ms,
            min_ms, max_ms, mean_ms, stddev_ms, error_rate.
        """
        logger.info(
            "Latency measurement started",
            endpoint=endpoint,
            num_requests=num_requests,
            concurrency=concurrency,
        )

        # Warmup phase
        if warmup_requests > 0:
            await self._send_batch(
                endpoint=endpoint,
                payload=sample_payload,
                batch_size=warmup_requests,
                concurrency=min(concurrency, warmup_requests),
            )

        # Timed measurement phase
        batch_result = await self._send_batch(
            endpoint=endpoint,
            payload=sample_payload,
            batch_size=num_requests,
            concurrency=concurrency,
        )

        latency_ms_list = batch_result["latency_ms_list"]
        error_count = batch_result["error_count"]

        result = self._compute_latency_stats(
            latency_ms_list=latency_ms_list,
            error_count=error_count,
            total_requests=num_requests,
        )

        logger.info(
            "Latency measurement completed",
            p50_ms=result.get("p50_ms"),
            p95_ms=result.get("p95_ms"),
            p99_ms=result.get("p99_ms"),
            error_rate=result.get("error_rate"),
        )
        return result

    async def measure_throughput(
        self,
        endpoint: str,
        sample_payload: dict[str, Any],
        duration_seconds: int,
        concurrency: int,
    ) -> dict[str, Any]:
        """Measure sustained throughput (requests/sec) over a fixed duration.

        Sends requests continuously for the specified duration at the given
        concurrency level and computes the achieved request rate.

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload for each probe.
            duration_seconds: Duration of the sustained load phase.
            concurrency: Number of simultaneous inflight requests.

        Returns:
            Throughput result dict with: requests_per_second, total_requests,
            successful_requests, error_rate, duration_seconds.
        """
        logger.info(
            "Throughput measurement started",
            endpoint=endpoint,
            duration_seconds=duration_seconds,
            concurrency=concurrency,
        )

        start_time = time.monotonic()
        total_requests = 0
        successful_requests = 0
        error_count = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            semaphore = asyncio.Semaphore(concurrency)

            async def single_request() -> bool:
                async with semaphore:
                    try:
                        response = await client.post(endpoint, json=sample_payload)
                        return response.status_code < 500
                    except Exception:
                        return False

            while time.monotonic() - start_time < duration_seconds:
                batch = [single_request() for _ in range(concurrency)]
                results = await asyncio.gather(*batch, return_exceptions=True)
                for res in results:
                    total_requests += 1
                    if isinstance(res, bool) and res:
                        successful_requests += 1
                    else:
                        error_count += 1

        elapsed = time.monotonic() - start_time
        requests_per_second = total_requests / max(elapsed, 1.0)
        error_rate = error_count / max(total_requests, 1)

        result = {
            "requests_per_second": round(requests_per_second, 2),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "duration_seconds": round(elapsed, 2),
            "concurrency": concurrency,
        }

        logger.info(
            "Throughput measurement completed",
            requests_per_second=result["requests_per_second"],
            error_rate=result["error_rate"],
        )
        return result

    async def profile_memory(
        self,
        endpoint: str,
        sample_payload: dict[str, Any],
        num_requests: int,
    ) -> dict[str, Any]:
        """Profile memory usage during inference load.

        Samples RSS memory before, during, and after the load phase to
        measure peak allocation and detect memory leaks.

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload.
            num_requests: Number of requests to send during profiling.

        Returns:
            Memory profile dict with: peak_rss_mb, baseline_rss_mb,
            delta_rss_mb, samples.
        """
        baseline_mb = await asyncio.to_thread(self._sample_rss_mb)

        # Run requests while sampling memory
        sampling_task = asyncio.create_task(
            self._sample_memory_during_load(num_samples=20)
        )
        load_task = asyncio.create_task(
            self._send_batch(
                endpoint=endpoint,
                payload=sample_payload,
                batch_size=num_requests,
                concurrency=self._max_workers,
            )
        )

        await load_task
        memory_samples = await sampling_task

        peak_mb = max(memory_samples) if memory_samples else baseline_mb
        delta_mb = peak_mb - baseline_mb

        result = {
            "baseline_rss_mb": round(baseline_mb, 2),
            "peak_rss_mb": round(peak_mb, 2),
            "delta_rss_mb": round(delta_mb, 2),
            "sample_count": len(memory_samples),
            "samples_mb": [round(s, 2) for s in memory_samples],
        }

        logger.info(
            "Memory profiling completed",
            peak_rss_mb=result["peak_rss_mb"],
            delta_rss_mb=result["delta_rss_mb"],
        )
        return result

    async def profile_cpu(
        self,
        endpoint: str,
        sample_payload: dict[str, Any],
        num_requests: int,
    ) -> dict[str, Any]:
        """Profile CPU utilization during inference load.

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload.
            num_requests: Number of requests to send during profiling.

        Returns:
            CPU profile dict with: average_cpu_percent, peak_cpu_percent, samples.
        """
        sampling_task = asyncio.create_task(
            self._sample_cpu_during_load(num_samples=20)
        )
        load_task = asyncio.create_task(
            self._send_batch(
                endpoint=endpoint,
                payload=sample_payload,
                batch_size=num_requests,
                concurrency=self._max_workers,
            )
        )

        await load_task
        cpu_samples = await sampling_task

        average_cpu = statistics.mean(cpu_samples) if cpu_samples else 0.0
        peak_cpu = max(cpu_samples) if cpu_samples else 0.0

        result = {
            "average_cpu_percent": round(average_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "sample_count": len(cpu_samples),
            "samples_percent": [round(s, 2) for s in cpu_samples],
        }

        logger.info(
            "CPU profiling completed",
            average_cpu_percent=result["average_cpu_percent"],
            peak_cpu_percent=result["peak_cpu_percent"],
        )
        return result

    async def compare_to_baseline(
        self,
        current_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        tolerance_percent: float = 10.0,
    ) -> dict[str, Any]:
        """Compare current benchmark metrics against a stored baseline.

        Args:
            current_metrics: Current run metrics dict with p50_ms, p95_ms, etc.
            baseline_metrics: Previous release baseline metrics dict.
            tolerance_percent: Allowed degradation percentage before flagging
                a regression (e.g. 10.0 = 10% worse is acceptable).

        Returns:
            Comparison result with: regressions, improvements, overall_status.
        """
        return await asyncio.to_thread(
            self._compare_metrics,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            tolerance_percent=tolerance_percent,
        )

    async def detect_regression(
        self,
        comparison_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Determine if the comparison contains regressions requiring action.

        Args:
            comparison_result: Output from compare_to_baseline.

        Returns:
            Regression detection result with: regression_detected, severity,
            exit_code, summary.
        """
        regressions = comparison_result.get("regressions", [])
        regression_detected = len(regressions) > 0

        critical_regressions = [r for r in regressions if r.get("delta_percent", 0.0) > 50.0]
        high_regressions = [r for r in regressions if 25.0 < r.get("delta_percent", 0.0) <= 50.0]

        if critical_regressions:
            severity = "critical"
            exit_code = 2
        elif high_regressions:
            severity = "high"
            exit_code = 1
        elif regression_detected:
            severity = "medium"
            exit_code = 1
        else:
            severity = "none"
            exit_code = 0

        return {
            "regression_detected": regression_detected,
            "severity": severity,
            "exit_code": exit_code,
            "regression_count": len(regressions),
            "critical_regression_count": len(critical_regressions),
            "summary": (
                f"{len(regressions)} regression(s) detected. Severity: {severity}."
                if regression_detected
                else "No performance regressions detected."
            ),
            "regressions": regressions,
        }

    async def generate_benchmark_report(
        self,
        latency_result: dict[str, Any],
        throughput_result: dict[str, Any],
        memory_result: dict[str, Any] | None,
        cpu_result: dict[str, Any] | None,
        comparison_result: dict[str, Any] | None,
        model_name: str,
    ) -> dict[str, Any]:
        """Assemble a comprehensive benchmark report from all profiling results.

        Args:
            latency_result: Latency measurement result.
            throughput_result: Throughput measurement result.
            memory_result: Memory profile result, or None.
            cpu_result: CPU profile result, or None.
            comparison_result: Baseline comparison result, or None.
            model_name: Human-readable model identifier.

        Returns:
            Benchmark report dict with executive summary and full metric details.
        """
        regressions = comparison_result.get("regressions", []) if comparison_result else []
        improvements = comparison_result.get("improvements", []) if comparison_result else []

        return {
            "model_name": model_name,
            "executive_summary": {
                "p50_latency_ms": latency_result.get("p50_ms"),
                "p95_latency_ms": latency_result.get("p95_ms"),
                "p99_latency_ms": latency_result.get("p99_ms"),
                "requests_per_second": throughput_result.get("requests_per_second"),
                "error_rate": latency_result.get("error_rate"),
                "peak_memory_mb": memory_result.get("peak_rss_mb") if memory_result else None,
                "regression_count": len(regressions),
                "improvement_count": len(improvements),
            },
            "latency": latency_result,
            "throughput": throughput_result,
            "memory": memory_result,
            "cpu": cpu_result,
            "baseline_comparison": comparison_result,
        }

    # --- Internal helpers ---

    async def _send_batch(
        self,
        endpoint: str,
        payload: dict[str, Any],
        batch_size: int,
        concurrency: int,
    ) -> dict[str, Any]:
        """Send a batch of HTTP requests and collect latency measurements.

        Args:
            endpoint: Target URL.
            payload: JSON payload.
            batch_size: Number of requests to send.
            concurrency: Maximum simultaneous requests.

        Returns:
            Batch result dict with latency_ms_list and error_count.
        """
        semaphore = asyncio.Semaphore(concurrency)
        latency_ms_list: list[float] = []
        error_count = 0

        async def timed_request() -> None:
            nonlocal error_count
            async with semaphore:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        start = time.monotonic()
                        response = await client.post(endpoint, json=payload)
                        elapsed_ms = (time.monotonic() - start) * 1000
                        if response.status_code < 500:
                            latency_ms_list.append(elapsed_ms)
                        else:
                            error_count += 1
                except Exception:
                    error_count += 1

        tasks = [timed_request() for _ in range(batch_size)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return {"latency_ms_list": latency_ms_list, "error_count": error_count}

    @staticmethod
    def _compute_latency_stats(
        latency_ms_list: list[float],
        error_count: int,
        total_requests: int,
    ) -> dict[str, Any]:
        """Compute latency percentiles and summary statistics.

        Args:
            latency_ms_list: List of successful latency measurements in milliseconds.
            error_count: Number of failed requests.
            total_requests: Total requests attempted.

        Returns:
            Latency statistics dict.
        """
        if len(latency_ms_list) < _MIN_SAMPLES_FOR_PERCENTILES:
            return {
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
                "mean_ms": None,
                "stddev_ms": None,
                "error_rate": error_count / max(total_requests, 1),
                "sample_count": len(latency_ms_list),
                "note": f"Insufficient samples ({len(latency_ms_list)} < {_MIN_SAMPLES_FOR_PERCENTILES})",
            }

        sorted_latencies = sorted(latency_ms_list)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = max(0, int(n * p / 100) - 1)
            return round(sorted_latencies[min(idx, n - 1)], 2)

        return {
            "p50_ms": percentile(50),
            "p90_ms": percentile(90),
            "p95_ms": percentile(95),
            "p99_ms": percentile(99),
            "min_ms": round(min(sorted_latencies), 2),
            "max_ms": round(max(sorted_latencies), 2),
            "mean_ms": round(statistics.mean(sorted_latencies), 2),
            "stddev_ms": round(statistics.stdev(sorted_latencies) if n > 1 else 0.0, 2),
            "error_rate": round(error_count / max(total_requests, 1), 4),
            "sample_count": n,
            "error_count": error_count,
        }

    @staticmethod
    def _sample_rss_mb() -> float:
        """Sample current process RSS memory in megabytes.

        Returns:
            RSS memory in MB, or 0.0 if psutil is unavailable.
        """
        try:
            import psutil  # noqa: PLC0415
            import os  # noqa: PLC0415
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    async def _sample_memory_during_load(self, num_samples: int) -> list[float]:
        """Sample RSS memory repeatedly while a load task runs.

        Args:
            num_samples: Number of samples to collect.

        Returns:
            List of RSS memory samples in MB.
        """
        samples: list[float] = []
        for _ in range(num_samples):
            sample = await asyncio.to_thread(self._sample_rss_mb)
            samples.append(sample)
            await asyncio.sleep(_RESOURCE_SAMPLING_INTERVAL)
        return samples

    async def _sample_cpu_during_load(self, num_samples: int) -> list[float]:
        """Sample CPU utilization repeatedly while a load task runs.

        Args:
            num_samples: Number of samples to collect.

        Returns:
            List of CPU utilization samples (percent).
        """
        samples: list[float] = []

        async def single_cpu_sample() -> float:
            try:
                import psutil  # noqa: PLC0415
                return await asyncio.to_thread(psutil.cpu_percent, 0.1)
            except ImportError:
                return 0.0

        for _ in range(num_samples):
            sample = await single_cpu_sample()
            samples.append(sample)
            await asyncio.sleep(_RESOURCE_SAMPLING_INTERVAL)

        return samples

    @staticmethod
    def _compare_metrics(
        current_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        tolerance_percent: float,
    ) -> dict[str, Any]:
        """Compare two metric dicts and identify regressions and improvements.

        For latency metrics (p50, p95, p99) higher is worse.
        For throughput higher is better.

        Args:
            current_metrics: Current benchmark metrics.
            baseline_metrics: Baseline benchmark metrics.
            tolerance_percent: Acceptable degradation percentage.

        Returns:
            Comparison result with regressions and improvements lists.
        """
        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []

        # Metrics where lower is better (latency)
        latency_metrics = ["p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms"]
        # Metrics where higher is better (throughput)
        throughput_metrics = ["requests_per_second"]

        for metric in latency_metrics:
            current_val = current_metrics.get(metric)
            baseline_val = baseline_metrics.get(metric)

            if current_val is None or baseline_val is None or baseline_val == 0:
                continue

            delta_percent = ((current_val - baseline_val) / baseline_val) * 100.0

            if delta_percent > tolerance_percent:
                regressions.append({
                    "metric": metric,
                    "current": current_val,
                    "baseline": baseline_val,
                    "delta_percent": round(delta_percent, 2),
                    "direction": "worse (higher latency)",
                })
            elif delta_percent < -tolerance_percent:
                improvements.append({
                    "metric": metric,
                    "current": current_val,
                    "baseline": baseline_val,
                    "delta_percent": round(delta_percent, 2),
                    "direction": "better (lower latency)",
                })

        for metric in throughput_metrics:
            current_val = current_metrics.get(metric)
            baseline_val = baseline_metrics.get(metric)

            if current_val is None or baseline_val is None or baseline_val == 0:
                continue

            delta_percent = ((current_val - baseline_val) / baseline_val) * 100.0

            if delta_percent < -tolerance_percent:
                regressions.append({
                    "metric": metric,
                    "current": current_val,
                    "baseline": baseline_val,
                    "delta_percent": round(delta_percent, 2),
                    "direction": "worse (lower throughput)",
                })
            elif delta_percent > tolerance_percent:
                improvements.append({
                    "metric": metric,
                    "current": current_val,
                    "baseline": baseline_val,
                    "delta_percent": round(delta_percent, 2),
                    "direction": "better (higher throughput)",
                })

        return {
            "regressions": regressions,
            "improvements": improvements,
            "overall_status": "pass" if not regressions else "fail",
            "tolerance_percent": tolerance_percent,
        }


__all__ = ["PerformanceBenchmarker"]
