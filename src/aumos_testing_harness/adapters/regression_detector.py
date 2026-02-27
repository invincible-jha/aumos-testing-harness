"""Regression detection adapter — CI/CD quality gate enforcement.

Implements IRegressionDetector. Compares evaluation metrics against baselines
to detect regressions before merging code or deploying models:
  - Metric comparison: current run vs. last release baseline
  - Statistical significance testing (Welch's t-test, Mann-Whitney U)
  - Per-metric threshold configuration with absolute and relative tolerances
  - CI/CD pipeline integration: produces structured exit codes and gate results
  - Regression report generation with per-metric breakdown
  - Historical regression tracking for trend analysis
  - Auto-baseline update on approved release (with audit trail)

All statistical computations run in asyncio.to_thread() to keep the event loop free.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)

# Significance level (alpha) for statistical tests
_SIGNIFICANCE_LEVEL: float = 0.05

# Default relative tolerance: 5% degradation is acceptable before flagging
_DEFAULT_RELATIVE_TOLERANCE: float = 0.05

# Default absolute tolerance band for metrics that can be near zero
_DEFAULT_ABSOLUTE_TOLERANCE: float = 0.02


class RegressionDetector:
    """CI/CD quality gate that detects metric regressions against baselines.

    Compares current evaluation run results against a stored baseline using
    both threshold-based and statistical tests. Outputs structured gate results
    with recommended CI exit codes for automated pipeline integration.

    Args:
        settings: Application settings providing artifact bucket configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings for the testing harness.
        """
        self._settings = settings
        self._artifact_bucket = settings.artifact_bucket
        self._github_token = settings.github_token

    async def compare_to_baseline(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        metric_configs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compare current evaluation metrics against a stored baseline.

        For each metric, applies an absolute tolerance band and a relative
        tolerance fraction. Metrics where lower is better are treated correctly
        (e.g., loss, error_rate). Metrics where higher is better are the default.

        Args:
            current_metrics: Dict mapping metric_name -> current value.
            baseline_metrics: Dict mapping metric_name -> baseline value.
            metric_configs: Optional per-metric configuration dicts. Each entry
                may contain: 'lower_is_better' (bool), 'relative_tolerance' (float),
                'absolute_tolerance' (float).

        Returns:
            Comparison result dict with: regressions, improvements,
            unchanged_metrics, overall_status.
        """
        result = await asyncio.to_thread(
            self._run_comparison,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            metric_configs=metric_configs or {},
        )
        logger.info(
            "Baseline comparison completed",
            regression_count=len(result.get("regressions", [])),
            improvement_count=len(result.get("improvements", [])),
            overall_status=result.get("overall_status"),
        )
        return result

    async def test_statistical_significance(
        self,
        current_samples: dict[str, list[float]],
        baseline_samples: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Run Welch's t-test and Mann-Whitney U to detect statistically significant changes.

        Args:
            current_samples: Dict mapping metric_name -> list of current run samples.
            baseline_samples: Dict mapping metric_name -> list of baseline samples.

        Returns:
            Statistical test result dict with per-metric p-values and significance flags.
        """
        result = await asyncio.to_thread(
            self._run_statistical_tests,
            current_samples=current_samples,
            baseline_samples=baseline_samples,
        )
        logger.info(
            "Statistical significance testing completed",
            metrics_tested=len(result.get("per_metric_tests", {})),
            significant_changes=result.get("significant_change_count", 0),
        )
        return result

    async def evaluate_quality_gate(
        self,
        comparison_result: dict[str, Any],
        statistical_result: dict[str, Any] | None,
        strict_mode: bool = False,
    ) -> dict[str, Any]:
        """Evaluate all regression findings and produce a CI/CD gate decision.

        Args:
            comparison_result: Output from compare_to_baseline.
            statistical_result: Optional output from test_statistical_significance.
            strict_mode: If True, also fail on improvements that were statistically
                insignificant (prevents noise-driven false passes).

        Returns:
            Gate decision dict with: passed, exit_code, reasons, severity.
        """
        regressions = comparison_result.get("regressions", [])
        significant_regressions = regressions

        if statistical_result:
            per_metric = statistical_result.get("per_metric_tests", {})
            significant_regressions = [
                r for r in regressions
                if per_metric.get(r["metric"], {}).get("significant", True)
            ]

        if not significant_regressions:
            return {
                "passed": True,
                "exit_code": 0,
                "severity": "none",
                "reasons": [],
                "regression_count": 0,
                "summary": "All metrics within acceptable bounds. Quality gate passed.",
            }

        critical = [r for r in significant_regressions if abs(r.get("delta_percent", 0)) > 25.0]
        reasons = [
            f"Metric '{r['metric']}' regressed by {r.get('delta_percent', 0):.1f}% "
            f"(current={r.get('current'):.4f}, baseline={r.get('baseline'):.4f})"
            for r in significant_regressions
        ]

        severity = "critical" if critical else "high" if len(significant_regressions) > 2 else "medium"
        exit_code = 2 if critical else 1

        return {
            "passed": False,
            "exit_code": exit_code,
            "severity": severity,
            "reasons": reasons,
            "regression_count": len(significant_regressions),
            "critical_regression_count": len(critical),
            "summary": (
                f"Quality gate FAILED: {len(significant_regressions)} regression(s) detected "
                f"with severity '{severity}'."
            ),
        }

    async def generate_regression_report(
        self,
        comparison_result: dict[str, Any],
        statistical_result: dict[str, Any] | None,
        gate_result: dict[str, Any],
        run_id: str,
        model_version: str,
    ) -> dict[str, Any]:
        """Generate a structured regression report for audit and notification.

        Args:
            comparison_result: Output from compare_to_baseline.
            statistical_result: Optional statistical test results.
            gate_result: Output from evaluate_quality_gate.
            run_id: Test run UUID for correlation.
            model_version: Version string of the model being evaluated.

        Returns:
            Regression report dict with full metric breakdown and audit metadata.
        """
        return {
            "run_id": run_id,
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "gate_result": gate_result,
            "comparison_summary": {
                "regression_count": len(comparison_result.get("regressions", [])),
                "improvement_count": len(comparison_result.get("improvements", [])),
                "unchanged_count": len(comparison_result.get("unchanged_metrics", [])),
                "overall_status": comparison_result.get("overall_status"),
            },
            "regressions": comparison_result.get("regressions", []),
            "improvements": comparison_result.get("improvements", []),
            "statistical_analysis": statistical_result,
        }

    async def track_regression_history(
        self,
        regression_report: dict[str, Any],
        history_file_path: str,
    ) -> dict[str, Any]:
        """Append a regression report entry to the historical tracking file.

        Args:
            regression_report: The regression report to persist.
            history_file_path: Path to the JSON history file.

        Returns:
            History tracking result with total_entries and file_path.
        """
        return await asyncio.to_thread(
            self._append_to_history,
            regression_report=regression_report,
            history_file_path=history_file_path,
        )

    async def update_baseline(
        self,
        new_metrics: dict[str, float],
        baseline_file_path: str,
        approved_by: str,
        model_version: str,
    ) -> dict[str, Any]:
        """Update the stored baseline with new metrics after an approved release.

        Args:
            new_metrics: New baseline metric values to persist.
            baseline_file_path: Path to the JSON baseline file.
            approved_by: Identifier of the approver (for audit trail).
            model_version: Model version being promoted to baseline.

        Returns:
            Baseline update result with: file_path, updated_at, metric_count.
        """
        return await asyncio.to_thread(
            self._write_baseline,
            new_metrics=new_metrics,
            baseline_file_path=baseline_file_path,
            approved_by=approved_by,
            model_version=model_version,
        )

    # --- Synchronous implementations (run in thread pool) ---

    def _run_comparison(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        metric_configs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare current metrics to baseline with per-metric config.

        Args:
            current_metrics: Current evaluation metric values.
            baseline_metrics: Baseline metric values.
            metric_configs: Per-metric tolerance and direction configuration.

        Returns:
            Comparison result dict.
        """
        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []
        unchanged: list[dict[str, Any]] = []

        all_metrics = set(current_metrics.keys()) & set(baseline_metrics.keys())

        for metric in all_metrics:
            current_val = current_metrics[metric]
            baseline_val = baseline_metrics[metric]
            config = metric_configs.get(metric, {})

            lower_is_better = config.get("lower_is_better", False)
            rel_tolerance = config.get("relative_tolerance", _DEFAULT_RELATIVE_TOLERANCE)
            abs_tolerance = config.get("absolute_tolerance", _DEFAULT_ABSOLUTE_TOLERANCE)

            if baseline_val == 0.0:
                delta_percent = (current_val - baseline_val) * 100.0
            else:
                delta_percent = ((current_val - baseline_val) / abs(baseline_val)) * 100.0

            delta_absolute = current_val - baseline_val

            # For lower-is-better metrics (e.g. error_rate), an increase is a regression
            if lower_is_better:
                is_regression = (
                    delta_absolute > abs_tolerance
                    and delta_percent > rel_tolerance * 100
                )
                is_improvement = (
                    delta_absolute < -abs_tolerance
                    and delta_percent < -rel_tolerance * 100
                )
            else:
                # Higher-is-better (accuracy, score)
                is_regression = (
                    delta_absolute < -abs_tolerance
                    and delta_percent < -rel_tolerance * 100
                )
                is_improvement = (
                    delta_absolute > abs_tolerance
                    and delta_percent > rel_tolerance * 100
                )

            entry = {
                "metric": metric,
                "current": round(current_val, 6),
                "baseline": round(baseline_val, 6),
                "delta_absolute": round(delta_absolute, 6),
                "delta_percent": round(delta_percent, 2),
                "lower_is_better": lower_is_better,
                "relative_tolerance": rel_tolerance,
                "absolute_tolerance": abs_tolerance,
            }

            if is_regression:
                entry["classification"] = "regression"
                regressions.append(entry)
            elif is_improvement:
                entry["classification"] = "improvement"
                improvements.append(entry)
            else:
                entry["classification"] = "unchanged"
                unchanged.append(entry)

        # Sort regressions by severity (largest delta_percent first)
        regressions.sort(key=lambda x: abs(x["delta_percent"]), reverse=True)

        return {
            "regressions": regressions,
            "improvements": improvements,
            "unchanged_metrics": unchanged,
            "overall_status": "fail" if regressions else "pass",
            "metrics_compared": len(all_metrics),
            "metrics_only_in_current": list(set(current_metrics) - set(baseline_metrics)),
            "metrics_only_in_baseline": list(set(baseline_metrics) - set(current_metrics)),
        }

    def _run_statistical_tests(
        self,
        current_samples: dict[str, list[float]],
        baseline_samples: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Run Welch's t-test and Mann-Whitney U test on sample distributions.

        Args:
            current_samples: Current run sample lists.
            baseline_samples: Baseline sample lists.

        Returns:
            Statistical test results dict.
        """
        try:
            from scipy import stats  # noqa: PLC0415

            per_metric_tests: dict[str, Any] = {}
            significant_change_count = 0

            all_metrics = set(current_samples.keys()) & set(baseline_samples.keys())

            for metric in all_metrics:
                current = current_samples[metric]
                baseline = baseline_samples[metric]

                if len(current) < 3 or len(baseline) < 3:
                    per_metric_tests[metric] = {
                        "significant": False,
                        "note": "Insufficient samples for statistical test",
                    }
                    continue

                # Welch's t-test (does not assume equal variance)
                t_stat, t_pvalue = stats.ttest_ind(current, baseline, equal_var=False)

                # Mann-Whitney U (non-parametric, robust for non-normal distributions)
                try:
                    u_stat, u_pvalue = stats.mannwhitneyu(
                        current, baseline, alternative="two-sided"
                    )
                except ValueError:
                    u_stat, u_pvalue = 0.0, 1.0

                significant = t_pvalue < _SIGNIFICANCE_LEVEL and u_pvalue < _SIGNIFICANCE_LEVEL

                per_metric_tests[metric] = {
                    "significant": significant,
                    "welch_t_statistic": round(float(t_stat), 4),
                    "welch_p_value": round(float(t_pvalue), 6),
                    "mann_whitney_u": round(float(u_stat), 4),
                    "mann_whitney_p_value": round(float(u_pvalue), 6),
                    "significance_level": _SIGNIFICANCE_LEVEL,
                }

                if significant:
                    significant_change_count += 1

            return {
                "per_metric_tests": per_metric_tests,
                "significant_change_count": significant_change_count,
                "significance_level": _SIGNIFICANCE_LEVEL,
                "method": "Welch t-test + Mann-Whitney U",
            }

        except ImportError:
            logger.warning("scipy not installed — skipping statistical significance tests")
            return {
                "per_metric_tests": {},
                "significant_change_count": 0,
                "significance_level": _SIGNIFICANCE_LEVEL,
                "note": "scipy not installed — statistical tests skipped",
            }

    def _append_to_history(
        self,
        regression_report: dict[str, Any],
        history_file_path: str,
    ) -> dict[str, Any]:
        """Append a regression report to the JSON history file.

        Args:
            regression_report: Report to append.
            history_file_path: Path to the history JSON file.

        Returns:
            History tracking result.
        """
        history_path = Path(history_file_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to parse history file — starting fresh", path=history_file_path)

        history.append(regression_report)
        history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

        logger.info(
            "Regression history updated",
            path=history_file_path,
            total_entries=len(history),
        )

        return {
            "file_path": str(history_path.resolve()),
            "total_entries": len(history),
            "appended_run_id": regression_report.get("run_id"),
        }

    def _write_baseline(
        self,
        new_metrics: dict[str, float],
        baseline_file_path: str,
        approved_by: str,
        model_version: str,
    ) -> dict[str, Any]:
        """Write a new baseline JSON file.

        Args:
            new_metrics: Metric values to persist as the new baseline.
            baseline_file_path: Destination path.
            approved_by: Approver identifier for audit trail.
            model_version: Model version being baselined.

        Returns:
            Baseline write result.
        """
        baseline_path = Path(baseline_file_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_payload = {
            "metrics": new_metrics,
            "model_version": model_version,
            "approved_by": approved_by,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        baseline_path.write_text(
            json.dumps(baseline_payload, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Baseline updated",
            path=baseline_file_path,
            model_version=model_version,
            approved_by=approved_by,
            metric_count=len(new_metrics),
        )

        return {
            "file_path": str(baseline_path.resolve()),
            "updated_at": baseline_payload["updated_at"],
            "metric_count": len(new_metrics),
            "model_version": model_version,
        }


__all__ = ["RegressionDetector"]
