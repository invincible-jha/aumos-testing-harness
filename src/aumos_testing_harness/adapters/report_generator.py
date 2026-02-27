"""Test report generator adapter — automated multi-format reporting.

Implements ITestReportGenerator. Aggregates all test type results into
unified reports with distribution, webhook delivery, and CI badge generation:
  - Test result aggregation across LLM, RAG, agent, red-team, adversarial,
    privacy, coverage, performance, and regression test types
  - Executive summary: pass/fail counts, aggregate scores, trend delta
  - Per-test-type detailed breakdown sections
  - Trend analysis: test health across the last N runs
  - JSON and PDF report formats (PDF via reportlab when available)
  - Badge generation in Shields.io URL format (pass/fail status shields)
  - Report distribution: webhook POST, optional email stub

All file I/O and report rendering are dispatched via asyncio.to_thread().
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)

# Shields.io badge base URL
_SHIELDS_BASE_URL = "https://img.shields.io/badge"

# Pass/fail colours for Shields.io
_BADGE_PASS_COLOR = "brightgreen"
_BADGE_FAIL_COLOR = "red"
_BADGE_WARN_COLOR = "yellow"


class TestReportGenerator:
    """Automated test report generator for all AumOS evaluation types.

    Aggregates results from LLM evaluation, RAG evaluation, agent evaluation,
    red-team assessment, adversarial testing, privacy testing, coverage analysis,
    and performance benchmarking into unified JSON or PDF reports.

    Args:
        settings: Application settings providing webhook secret and artifact bucket.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings for the testing harness.
        """
        self._settings = settings
        self._webhook_secret = settings.webhook_secret
        self._artifact_bucket = settings.artifact_bucket

    async def aggregate_results(
        self,
        run_id: str,
        llm_results: list[dict[str, Any]] | None = None,
        rag_results: list[dict[str, Any]] | None = None,
        agent_results: list[dict[str, Any]] | None = None,
        red_team_results: list[dict[str, Any]] | None = None,
        adversarial_results: list[dict[str, Any]] | None = None,
        privacy_results: list[dict[str, Any]] | None = None,
        coverage_result: dict[str, Any] | None = None,
        performance_result: dict[str, Any] | None = None,
        regression_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregate all test results from all test types into a unified structure.

        Args:
            run_id: Test run UUID for correlation.
            llm_results: LLM evaluation metric results.
            rag_results: RAG evaluation metric results.
            agent_results: Agent evaluation metric results.
            red_team_results: Red-team probe results.
            adversarial_results: Adversarial robustness test results.
            privacy_results: Privacy attack test results.
            coverage_result: Code/input space coverage result.
            performance_result: Performance benchmark result.
            regression_result: Regression detection gate result.

        Returns:
            Aggregated report payload with executive summary and per-type details.
        """
        return await asyncio.to_thread(
            self._build_aggregated_report,
            run_id=run_id,
            llm_results=llm_results or [],
            rag_results=rag_results or [],
            agent_results=agent_results or [],
            red_team_results=red_team_results or [],
            adversarial_results=adversarial_results or [],
            privacy_results=privacy_results or [],
            coverage_result=coverage_result,
            performance_result=performance_result,
            regression_result=regression_result,
        )

    async def generate_executive_summary(
        self,
        aggregated_report: dict[str, Any],
        model_name: str,
        ci_build_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a concise executive summary from the aggregated report.

        Args:
            aggregated_report: Output from aggregate_results.
            model_name: Human-readable model identifier.
            ci_build_id: Optional CI build identifier for correlation.

        Returns:
            Executive summary dict with top-level pass/fail status and key metrics.
        """
        return await asyncio.to_thread(
            self._build_executive_summary,
            aggregated_report=aggregated_report,
            model_name=model_name,
            ci_build_id=ci_build_id,
        )

    async def analyze_trend(
        self,
        historical_reports: list[dict[str, Any]],
        current_report: dict[str, Any],
        window_size: int = 10,
    ) -> dict[str, Any]:
        """Analyze test health trend over the last window_size runs.

        Args:
            historical_reports: Ordered list of past aggregated reports (oldest first).
            current_report: Current run's aggregated report.
            window_size: Number of historical runs to include in the trend window.

        Returns:
            Trend analysis dict with moving averages, direction, and alerts.
        """
        return await asyncio.to_thread(
            self._compute_trend,
            historical_reports=historical_reports,
            current_report=current_report,
            window_size=window_size,
        )

    async def generate_report(
        self,
        aggregated_report: dict[str, Any],
        output_format: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Render and write a report file in the specified format.

        Args:
            aggregated_report: Full aggregated report payload.
            output_format: 'json' or 'pdf'.
            output_path: Destination file path.

        Returns:
            Report generation result with output_path, format, and size_bytes.
        """
        return await asyncio.to_thread(
            self._write_report_file,
            aggregated_report=aggregated_report,
            output_format=output_format,
            output_path=output_path,
        )

    async def generate_badge(
        self,
        aggregated_report: dict[str, Any],
        badge_type: str = "status",
    ) -> dict[str, Any]:
        """Generate a Shields.io badge URL for the test run result.

        Args:
            aggregated_report: Aggregated report payload.
            badge_type: Badge type: 'status' (pass/fail), 'score' (numeric),
                or 'coverage' (coverage percentage).

        Returns:
            Badge dict with: url, label, message, color, markdown.
        """
        return await asyncio.to_thread(
            self._build_badge,
            aggregated_report=aggregated_report,
            badge_type=badge_type,
        )

    async def distribute_via_webhook(
        self,
        aggregated_report: dict[str, Any],
        webhook_url: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST the aggregated report to a webhook endpoint.

        Computes an HMAC-SHA256 signature of the payload body using the
        configured webhook_secret and attaches it as X-AumOS-Signature.

        Args:
            aggregated_report: Report payload to deliver.
            webhook_url: Destination webhook URL.
            headers: Optional additional HTTP headers.

        Returns:
            Delivery result with: status_code, success, response_ms.
        """
        import hashlib  # noqa: PLC0415
        import hmac  # noqa: PLC0415
        import time  # noqa: PLC0415

        payload_bytes = json.dumps(aggregated_report, default=str).encode("utf-8")
        signature = ""

        if self._webhook_secret:
            signature = hmac.new(
                self._webhook_secret.encode("utf-8"),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()

        delivery_headers = {
            "Content-Type": "application/json",
            "X-AumOS-Signature": f"sha256={signature}",
            "X-AumOS-Event": "test.report.completed",
        }
        if headers:
            delivery_headers.update(headers)

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    content=payload_bytes,
                    headers=delivery_headers,
                )
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            success = response.status_code < 400

            logger.info(
                "Webhook delivery completed",
                webhook_url=webhook_url,
                status_code=response.status_code,
                success=success,
                elapsed_ms=elapsed_ms,
            )
            return {
                "success": success,
                "status_code": response.status_code,
                "response_ms": elapsed_ms,
                "webhook_url": webhook_url,
            }
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            logger.error("Webhook delivery failed", webhook_url=webhook_url, error=str(exc))
            return {
                "success": False,
                "status_code": None,
                "response_ms": elapsed_ms,
                "webhook_url": webhook_url,
                "error": str(exc),
            }

    # --- Synchronous implementations (run in thread pool) ---

    def _build_aggregated_report(
        self,
        run_id: str,
        llm_results: list[dict[str, Any]],
        rag_results: list[dict[str, Any]],
        agent_results: list[dict[str, Any]],
        red_team_results: list[dict[str, Any]],
        adversarial_results: list[dict[str, Any]],
        privacy_results: list[dict[str, Any]],
        coverage_result: dict[str, Any] | None,
        performance_result: dict[str, Any] | None,
        regression_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the full aggregated report structure.

        Args:
            run_id: Test run identifier.
            llm_results: LLM metric results.
            rag_results: RAG metric results.
            agent_results: Agent metric results.
            red_team_results: Red-team probe results.
            adversarial_results: Adversarial test results.
            privacy_results: Privacy test results.
            coverage_result: Coverage result.
            performance_result: Performance result.
            regression_result: Regression gate result.

        Returns:
            Full aggregated report dict.
        """
        all_metric_results = llm_results + rag_results + agent_results

        total_tests = len(all_metric_results)
        passed_tests = sum(1 for r in all_metric_results if r.get("passed", False))
        failed_tests = total_tests - passed_tests
        aggregate_score = (
            sum(r.get("score", 0.0) for r in all_metric_results) / max(total_tests, 1)
        )

        # Red-team aggregate
        red_team_critical = sum(
            1 for r in red_team_results if r.get("success_rate", 0.0) > 0.5
        )

        # Adversarial aggregate
        adversarial_failures = sum(
            1 for r in adversarial_results if not r.get("passed", True)
        )

        # Privacy aggregate
        privacy_risks = sum(
            1 for r in privacy_results
            if r.get("vulnerability_level") in ("critical", "high")
        )

        # Coverage aggregate
        line_coverage = coverage_result.get("line_coverage") if coverage_result else None

        # Performance aggregate (from executive_summary if nested)
        p95_ms: float | None = None
        if performance_result:
            exec_summary = performance_result.get("executive_summary", performance_result)
            p95_ms = exec_summary.get("p95_latency_ms")

        # Regression gate
        regression_passed = (
            regression_result.get("passed", True) if regression_result else True
        )

        overall_pass = (
            passed_tests == total_tests
            and red_team_critical == 0
            and adversarial_failures == 0
            and privacy_risks == 0
            and regression_passed
        )

        return {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "overall_pass": overall_pass,
            "aggregate_score": round(aggregate_score, 4),
            "total_metric_tests": total_tests,
            "passed_metric_tests": passed_tests,
            "failed_metric_tests": failed_tests,
            "pass_rate": round(passed_tests / max(total_tests, 1), 4),
            "sections": {
                "llm_evaluation": {
                    "results": llm_results,
                    "total": len(llm_results),
                    "passed": sum(1 for r in llm_results if r.get("passed", False)),
                },
                "rag_evaluation": {
                    "results": rag_results,
                    "total": len(rag_results),
                    "passed": sum(1 for r in rag_results if r.get("passed", False)),
                },
                "agent_evaluation": {
                    "results": agent_results,
                    "total": len(agent_results),
                    "passed": sum(1 for r in agent_results if r.get("passed", False)),
                },
                "red_team": {
                    "results": red_team_results,
                    "critical_count": red_team_critical,
                    "total_categories": len(red_team_results),
                },
                "adversarial": {
                    "results": adversarial_results,
                    "failure_count": adversarial_failures,
                    "total": len(adversarial_results),
                },
                "privacy": {
                    "results": privacy_results,
                    "high_risk_count": privacy_risks,
                    "total": len(privacy_results),
                },
                "coverage": coverage_result,
                "performance": {
                    "p95_latency_ms": p95_ms,
                    "full_result": performance_result,
                },
                "regression": regression_result,
            },
        }

    def _build_executive_summary(
        self,
        aggregated_report: dict[str, Any],
        model_name: str,
        ci_build_id: str | None,
    ) -> dict[str, Any]:
        """Build the executive summary section.

        Args:
            aggregated_report: Full aggregated report.
            model_name: Model identifier.
            ci_build_id: Optional CI build ID.

        Returns:
            Executive summary dict.
        """
        sections = aggregated_report.get("sections", {})
        coverage = sections.get("coverage", {})
        perf = sections.get("performance", {})

        return {
            "model_name": model_name,
            "ci_build_id": ci_build_id,
            "generated_at": aggregated_report.get("generated_at"),
            "overall_pass": aggregated_report.get("overall_pass", False),
            "overall_status": "PASS" if aggregated_report.get("overall_pass") else "FAIL",
            "aggregate_score": aggregated_report.get("aggregate_score"),
            "pass_rate": aggregated_report.get("pass_rate"),
            "total_metric_tests": aggregated_report.get("total_metric_tests"),
            "passed_metric_tests": aggregated_report.get("passed_metric_tests"),
            "failed_metric_tests": aggregated_report.get("failed_metric_tests"),
            "red_team_critical_count": sections.get("red_team", {}).get("critical_count", 0),
            "adversarial_failure_count": sections.get("adversarial", {}).get("failure_count", 0),
            "privacy_high_risk_count": sections.get("privacy", {}).get("high_risk_count", 0),
            "line_coverage": coverage.get("line_coverage") if coverage else None,
            "p95_latency_ms": perf.get("p95_latency_ms"),
            "regression_passed": sections.get("regression", {}).get("passed", True) if sections.get("regression") else None,
        }

    def _compute_trend(
        self,
        historical_reports: list[dict[str, Any]],
        current_report: dict[str, Any],
        window_size: int,
    ) -> dict[str, Any]:
        """Compute test health trend metrics over the historical window.

        Args:
            historical_reports: Past aggregated reports (oldest first).
            current_report: Current aggregated report.
            window_size: Number of runs to include.

        Returns:
            Trend analysis dict.
        """
        window = (historical_reports[-window_size:] if len(historical_reports) > window_size
                  else historical_reports)

        historical_scores = [r.get("aggregate_score", 0.0) for r in window]
        historical_pass_rates = [r.get("pass_rate", 0.0) for r in window]

        current_score = current_report.get("aggregate_score", 0.0)
        current_pass_rate = current_report.get("pass_rate", 0.0)

        all_scores = historical_scores + [current_score]
        all_pass_rates = historical_pass_rates + [current_pass_rate]

        score_moving_avg = sum(all_scores) / max(len(all_scores), 1)
        pass_rate_moving_avg = sum(all_pass_rates) / max(len(all_pass_rates), 1)

        if historical_scores:
            last_score = historical_scores[-1]
            score_delta = current_score - last_score
            trend_direction = (
                "improving" if score_delta > 0.01
                else "declining" if score_delta < -0.01
                else "stable"
            )
        else:
            score_delta = 0.0
            trend_direction = "stable"

        # Alert if declining trend persists for last 3 runs
        declining_streak = 0
        if len(historical_scores) >= 2:
            for i in range(len(historical_scores) - 1, max(len(historical_scores) - 4, -1), -1):
                if i > 0 and historical_scores[i] < historical_scores[i - 1]:
                    declining_streak += 1
                else:
                    break

        alerts: list[str] = []
        if declining_streak >= 2:
            alerts.append(f"Test health declining for {declining_streak + 1} consecutive runs.")
        if current_pass_rate < 0.7:
            alerts.append(f"Current pass rate {current_pass_rate:.0%} is below 70% threshold.")

        return {
            "window_size": len(all_scores),
            "current_score": round(current_score, 4),
            "score_delta_from_last": round(score_delta, 4),
            "score_moving_average": round(score_moving_avg, 4),
            "pass_rate_moving_average": round(pass_rate_moving_avg, 4),
            "trend_direction": trend_direction,
            "declining_streak": declining_streak,
            "alerts": alerts,
        }

    def _write_report_file(
        self,
        aggregated_report: dict[str, Any],
        output_format: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Write report to disk in the requested format.

        Args:
            aggregated_report: Report payload.
            output_format: 'json' or 'pdf'.
            output_path: Destination path.

        Returns:
            Write result dict.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            content = json.dumps(aggregated_report, indent=2, default=str)
            output_file.write_text(content, encoding="utf-8")

        elif output_format == "pdf":
            self._write_pdf_report(aggregated_report, output_file)

        else:
            raise ValueError(f"Unsupported format: '{output_format}'. Use 'json' or 'pdf'.")

        size_bytes = output_file.stat().st_size
        logger.info(
            "Report written",
            output_format=output_format,
            output_path=output_path,
            size_bytes=size_bytes,
        )

        return {
            "output_path": str(output_file.resolve()),
            "format": output_format,
            "size_bytes": size_bytes,
        }

    def _write_pdf_report(
        self,
        aggregated_report: dict[str, Any],
        output_file: Path,
    ) -> None:
        """Write a PDF report using reportlab when available.

        Falls back to a JSON-in-PDF wrapper if reportlab is not installed.

        Args:
            aggregated_report: Report payload.
            output_file: Destination path.
        """
        try:
            from reportlab.lib.pagesizes import letter  # noqa: PLC0415
            from reportlab.lib.styles import getSampleStyleSheet  # noqa: PLC0415
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # noqa: PLC0415

            doc = SimpleDocTemplate(str(output_file), pagesize=letter)
            styles = getSampleStyleSheet()
            story: list[Any] = []

            overall_status = "PASS" if aggregated_report.get("overall_pass") else "FAIL"
            story.append(Paragraph("AumOS Test Harness Report", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Run ID: {aggregated_report.get('run_id', 'N/A')}", styles["Normal"]))
            story.append(Paragraph(f"Generated: {aggregated_report.get('generated_at', 'N/A')}", styles["Normal"]))
            story.append(Paragraph(f"Overall Status: {overall_status}", styles["Heading2"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Aggregate Score: {aggregated_report.get('aggregate_score', 0.0):.2%}",
                styles["Normal"],
            ))
            story.append(Paragraph(
                f"Pass Rate: {aggregated_report.get('pass_rate', 0.0):.2%}",
                styles["Normal"],
            ))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Full report data available in JSON format.", styles["Normal"]))

            doc.build(story)

        except ImportError:
            logger.warning("reportlab not installed — writing JSON-formatted PDF placeholder")
            # Write a minimal text file with .pdf extension as fallback
            output_file.write_text(
                json.dumps(aggregated_report, indent=2, default=str),
                encoding="utf-8",
            )

    def _build_badge(
        self,
        aggregated_report: dict[str, Any],
        badge_type: str,
    ) -> dict[str, Any]:
        """Build a Shields.io badge URL for the test result.

        Args:
            aggregated_report: Aggregated report payload.
            badge_type: 'status', 'score', or 'coverage'.

        Returns:
            Badge dict with url, label, message, color, markdown.
        """
        if badge_type == "status":
            overall_pass = aggregated_report.get("overall_pass", False)
            label = "AumOS%20Tests"
            message = "passing" if overall_pass else "failing"
            color = _BADGE_PASS_COLOR if overall_pass else _BADGE_FAIL_COLOR

        elif badge_type == "score":
            score = aggregated_report.get("aggregate_score", 0.0)
            label = "eval%20score"
            message = f"{score:.0%}"
            color = _BADGE_PASS_COLOR if score >= 0.7 else _BADGE_WARN_COLOR if score >= 0.5 else _BADGE_FAIL_COLOR

        elif badge_type == "coverage":
            coverage = aggregated_report.get("sections", {}).get("coverage", {})
            line_cov = coverage.get("line_coverage") if coverage else None
            label = "coverage"
            if line_cov is not None:
                message = f"{line_cov:.0%}"
                color = _BADGE_PASS_COLOR if line_cov >= 0.8 else _BADGE_WARN_COLOR if line_cov >= 0.5 else _BADGE_FAIL_COLOR
            else:
                message = "unknown"
                color = "lightgrey"

        else:
            label = "AumOS"
            message = "unknown"
            color = "lightgrey"

        url = f"{_SHIELDS_BASE_URL}/{label}-{message}-{color}"

        return {
            "url": url,
            "label": label.replace("%20", " "),
            "message": message,
            "color": color,
            "markdown": f"![{label.replace('%20', ' ')}]({url})",
        }


__all__ = ["TestReportGenerator"]
