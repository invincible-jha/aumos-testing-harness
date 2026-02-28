"""Custom metric plugin runner for aumos-testing-harness.

Allows users to register and execute custom evaluation metrics that implement
a standard protocol. Metrics are sandboxed and stored in the thr_custom_metrics
database table.

GAP-177: Custom Metric Plugins
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@runtime_checkable
class CustomMetricProtocol(Protocol):
    """Protocol that all custom evaluation metrics must implement.

    Custom metrics are callables that accept an input/output pair and
    return a normalized float score in [0, 1] plus optional metadata.
    """

    @property
    def name(self) -> str:
        """Unique metric identifier."""
        ...

    @property
    def threshold(self) -> float:
        """Minimum passing score (default 0.5)."""
        ...

    async def measure(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single input/output pair.

        Args:
            input_text: The user query or prompt.
            actual_output: The model's response to evaluate.
            expected_output: Ground-truth answer for comparison (optional).
            context: Retrieved context chunks (for RAG metrics).

        Returns:
            Dict with at minimum 'score' (float in [0,1]) and 'passed' (bool).
        """
        ...


class CustomMetricRunner:
    """Manages registration and execution of custom evaluation metric plugins.

    Maintains an in-memory registry of custom metric implementations and
    dispatches evaluation requests to them asynchronously.

    Args:
        timeout_seconds: Maximum time to wait for a single metric evaluation.
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        """Initialise the custom metric runner.

        Args:
            timeout_seconds: Per-metric evaluation timeout.
        """
        self._registry: dict[str, CustomMetricProtocol] = {}
        self._timeout = timeout_seconds

    def register(self, metric: CustomMetricProtocol) -> None:
        """Register a custom metric plugin.

        Args:
            metric: Metric implementation satisfying CustomMetricProtocol.

        Raises:
            ValueError: If a metric with the same name is already registered.
        """
        if metric.name in self._registry:
            raise ValueError(f"Metric '{metric.name}' is already registered")
        self._registry[metric.name] = metric
        logger.info("custom_metric_registered", name=metric.name)

    def unregister(self, metric_name: str) -> None:
        """Remove a metric plugin from the registry.

        Args:
            metric_name: Name of the metric to remove.
        """
        self._registry.pop(metric_name, None)
        logger.info("custom_metric_unregistered", name=metric_name)

    def list_metrics(self) -> list[dict[str, Any]]:
        """List all registered custom metrics.

        Returns:
            List of dicts with 'name' and 'threshold' for each registered metric.
        """
        return [{"name": m.name, "threshold": m.threshold} for m in self._registry.values()]

    async def run_all(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run all registered metrics against a single test case.

        Args:
            input_text: Query or prompt being evaluated.
            actual_output: Model response to score.
            expected_output: Ground-truth answer (optional).
            context: Retrieved context for RAG metrics (optional).

        Returns:
            List of result dicts, one per registered metric.
        """
        results: list[dict[str, Any]] = []
        for metric_name, metric in self._registry.items():
            try:
                result = await asyncio.wait_for(
                    metric.measure(
                        input_text=input_text,
                        actual_output=actual_output,
                        expected_output=expected_output,
                        context=context,
                    ),
                    timeout=self._timeout,
                )
                result["metric_name"] = metric_name
                result["passed"] = result.get("score", 0.0) >= metric.threshold
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning("custom_metric_timeout", metric=metric_name, timeout=self._timeout)
                results.append({
                    "metric_name": metric_name,
                    "score": 0.0,
                    "passed": False,
                    "error": f"Timed out after {self._timeout}s",
                })
            except Exception as exc:
                logger.error("custom_metric_error", metric=metric_name, error=str(exc))
                results.append({
                    "metric_name": metric_name,
                    "score": 0.0,
                    "passed": False,
                    "error": str(exc),
                })
        return results

    async def run_single(
        self,
        metric_name: str,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single named metric against a test case.

        Args:
            metric_name: Name of the metric to run.
            input_text: Query or prompt.
            actual_output: Model response.
            expected_output: Optional ground truth.
            context: Optional retrieved context.

        Returns:
            Result dict from the metric.

        Raises:
            KeyError: If the metric is not registered.
        """
        if metric_name not in self._registry:
            raise KeyError(f"Metric '{metric_name}' is not registered")
        metric = self._registry[metric_name]
        result = await asyncio.wait_for(
            metric.measure(
                input_text=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context,
            ),
            timeout=self._timeout,
        )
        result["metric_name"] = metric_name
        result["passed"] = result.get("score", 0.0) >= metric.threshold
        return result
