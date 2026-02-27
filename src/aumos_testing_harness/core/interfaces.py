"""Abstract interfaces (Protocol classes) for aumos-testing-harness.

Defining interfaces as Protocol classes enables:
  - Dependency injection in services (services depend on interfaces, not implementations)
  - Easy mocking in tests without subclassing
  - Clear contracts between layers

Services depend on these protocols; adapter implementations satisfy them.
"""

import uuid
from typing import Protocol, runtime_checkable

from aumos_common.auth import TenantContext
from aumos_common.pagination import PageRequest, PageResponse

from aumos_testing_harness.core.models import (
    RedTeamAttackType,
    RedTeamReport,
    RunStatus,
    TestResult,
    TestRun,
    TestSuite,
)


@runtime_checkable
class ITestSuiteRepository(Protocol):
    """Repository interface for TestSuite persistence."""

    async def get_by_id(self, suite_id: uuid.UUID, tenant: TenantContext) -> TestSuite | None:
        """Retrieve a TestSuite by its primary key within tenant scope.

        Args:
            suite_id: The UUID of the test suite.
            tenant: Tenant context for RLS isolation.

        Returns:
            The matching TestSuite, or None if not found.
        """
        ...

    async def list_all(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestSuite]:
        """List all test suites for a tenant, paginated.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestSuite records.
        """
        ...

    async def create(
        self,
        tenant: TenantContext,
        name: str,
        suite_type: str,
        config: dict,
        description: str | None = None,
    ) -> TestSuite:
        """Persist a new TestSuite.

        Args:
            tenant: Tenant context for RLS isolation.
            name: Human-readable suite name.
            suite_type: Evaluation domain (llm/rag/agent/red-team).
            config: Suite configuration JSONB payload.
            description: Optional human-readable description.

        Returns:
            The newly created TestSuite.
        """
        ...

    async def delete(self, suite_id: uuid.UUID, tenant: TenantContext) -> None:
        """Soft-delete a TestSuite.

        Args:
            suite_id: The UUID of the suite to delete.
            tenant: Tenant context for RLS isolation.
        """
        ...


@runtime_checkable
class ITestRunRepository(Protocol):
    """Repository interface for TestRun persistence."""

    async def get_by_id(self, run_id: uuid.UUID, tenant: TenantContext) -> TestRun | None:
        """Retrieve a TestRun by its primary key.

        Args:
            run_id: The UUID of the test run.
            tenant: Tenant context for RLS isolation.

        Returns:
            The matching TestRun, or None if not found.
        """
        ...

    async def list_all(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestRun]:
        """List all test runs for a tenant, paginated.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestRun records.
        """
        ...

    async def create(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        ci_build_id: str | None = None,
    ) -> TestRun:
        """Create a new TestRun record in PENDING status.

        Args:
            tenant: Tenant context for RLS isolation.
            suite_id: The suite being executed.
            ci_build_id: Optional CI build identifier for correlation.

        Returns:
            The newly created TestRun.
        """
        ...

    async def update_status(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
        status: RunStatus,
        summary: dict | None = None,
    ) -> TestRun:
        """Update the status (and optional summary) of an existing TestRun.

        Args:
            run_id: The UUID of the run to update.
            tenant: Tenant context for RLS isolation.
            status: New lifecycle status.
            summary: Optional aggregated scoring summary to store.

        Returns:
            The updated TestRun.
        """
        ...


@runtime_checkable
class ITestResultRepository(Protocol):
    """Repository interface for TestResult persistence."""

    async def list_by_run(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
        page: PageRequest,
    ) -> PageResponse[TestResult]:
        """List all test results for a given run, paginated.

        Args:
            run_id: The UUID of the parent test run.
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestResult records.
        """
        ...

    async def bulk_create(
        self,
        tenant: TenantContext,
        run_id: uuid.UUID,
        results: list[dict],
    ) -> list[TestResult]:
        """Persist multiple TestResult rows in a single transaction.

        Args:
            tenant: Tenant context for RLS isolation.
            run_id: The parent test run UUID.
            results: List of result dicts with keys: metric_name, score, threshold, passed, details.

        Returns:
            List of created TestResult records.
        """
        ...


@runtime_checkable
class IRedTeamReportRepository(Protocol):
    """Repository interface for RedTeamReport persistence."""

    async def get_by_run(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
    ) -> list[RedTeamReport]:
        """Retrieve all red-team reports for a run.

        Args:
            run_id: The UUID of the parent test run.
            tenant: Tenant context for RLS isolation.

        Returns:
            List of RedTeamReport records for the run.
        """
        ...

    async def create(
        self,
        tenant: TenantContext,
        run_id: uuid.UUID,
        attack_type: RedTeamAttackType,
        success_rate: float,
        vulnerabilities: dict,
        total_probes: int,
        successful_attacks: int,
    ) -> RedTeamReport:
        """Persist a new RedTeamReport.

        Args:
            tenant: Tenant context for RLS isolation.
            run_id: Parent test run UUID.
            attack_type: OWASP LLM attack category.
            success_rate: Fraction of probes that succeeded (0.0 = fully resilient).
            vulnerabilities: Sanitised vulnerability descriptions (no raw attack prompts).
            total_probes: Total number of attack probes attempted.
            successful_attacks: Number of successful attack probes.

        Returns:
            The newly created RedTeamReport.
        """
        ...


@runtime_checkable
class ILLMEvaluator(Protocol):
    """Interface for LLM evaluation backend (deepeval implementation)."""

    async def evaluate(
        self,
        metric_names: list[str],
        test_cases: list[dict],
        threshold: float,
    ) -> list[dict]:
        """Run one or more LLM metrics against a batch of test cases.

        Args:
            metric_names: Names of metrics from MetricName enum to compute.
            test_cases: List of test case dicts with input, expected_output, context.
            threshold: Pass/fail threshold (0.0-1.0).

        Returns:
            List of result dicts with: metric_name, score, passed, details.
        """
        ...


@runtime_checkable
class IRAGEvaluator(Protocol):
    """Interface for RAG evaluation backend (RAGAS implementation)."""

    async def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None,
    ) -> list[dict]:
        """Evaluate a RAG pipeline using RAGAS metrics.

        Args:
            questions: User queries.
            answers: Generated answers from the RAG pipeline.
            contexts: Retrieved document chunks per question.
            ground_truths: Optional reference answers for supervised metrics.

        Returns:
            List of result dicts with: metric_name, score, passed, details.
        """
        ...


@runtime_checkable
class IAgentEvaluator(Protocol):
    """Interface for agent capability evaluation backend."""

    async def evaluate(
        self,
        task_definitions: list[dict],
        agent_trajectories: list[dict],
        threshold: float,
    ) -> list[dict]:
        """Evaluate agent task performance against recorded trajectories.

        Args:
            task_definitions: Goal criteria and expected tool usage.
            agent_trajectories: Recorded step-by-step agent execution.
            threshold: Pass/fail threshold.

        Returns:
            List of result dicts with: metric_name, score, passed, details.
        """
        ...


@runtime_checkable
class IRedTeamRunner(Protocol):
    """Interface for OWASP red-team probe runner (Garak + Giskard)."""

    async def run_probes(
        self,
        target_endpoint: str,
        attack_types: list[RedTeamAttackType],
        max_attempts: int,
    ) -> list[dict]:
        """Execute red-team probes against a target model endpoint.

        Args:
            target_endpoint: URL of the model inference endpoint to test.
            attack_types: OWASP LLM attack categories to probe.
            max_attempts: Maximum probe attempts per attack category.

        Returns:
            List of report dicts with: attack_type, success_rate, vulnerabilities,
            total_probes, successful_attacks.
        """
        ...


@runtime_checkable
class IAdversarialTester(Protocol):
    """Interface for input perturbation robustness testing."""

    async def run_text_perturbation(
        self,
        test_cases: list[dict],
        perturbation_types: list[str],
        threshold: float,
    ) -> list[dict]:
        """Test model robustness against text perturbations (typos, synonyms, paraphrase).

        Args:
            test_cases: List of dicts with 'input' and 'expected_output'.
            perturbation_types: Subset of ['typo', 'synonym', 'paraphrase'].
            threshold: Minimum robustness score to pass (0.0-1.0).

        Returns:
            List of result dicts with: perturbation_type, robustness_score,
            attack_success_rate, passed, details.
        """
        ...

    async def run_adversarial_examples(
        self,
        model_endpoint: str,
        test_cases: list[dict],
        epsilon: float,
        threshold: float,
    ) -> list[dict]:
        """Generate FGSM-style adversarial examples and measure attack success rate.

        Args:
            model_endpoint: URL of the model inference endpoint.
            test_cases: List of dicts with 'input' and 'expected_label'.
            epsilon: Perturbation magnitude (0.0-1.0).
            threshold: Minimum robustness score to pass.

        Returns:
            List of result dicts with attack_success_rate, robustness_score, passed.
        """
        ...

    async def generate_vulnerability_report(
        self,
        all_results: list[dict],
        model_name: str,
    ) -> dict:
        """Aggregate perturbation results into a vulnerability report.

        Args:
            all_results: All perturbation result dicts from all test types.
            model_name: Human-readable model identifier.

        Returns:
            Vulnerability report dict with executive summary and per-vector details.
        """
        ...


@runtime_checkable
class IPrivacyTester(Protocol):
    """Interface for privacy attack simulation (membership and attribute inference)."""

    async def run_membership_inference_attack(
        self,
        model_endpoint: str,
        member_records: list[dict],
        non_member_records: list[dict],
        membership_threshold: float,
    ) -> dict:
        """Execute membership inference attack via confidence thresholding.

        Args:
            model_endpoint: URL of the target model endpoint.
            member_records: Records known to be in the training set.
            non_member_records: Records known to be out of the training set.
            membership_threshold: Confidence score above which records are members.

        Returns:
            Attack result dict with attack_accuracy, advantage, auc_roc,
            per_record_scores, vulnerability_level.
        """
        ...

    async def verify_differential_privacy(
        self,
        epsilon: float,
        delta: float,
        mechanism: str,
        sensitivity: float,
        noise_scale: float,
    ) -> dict:
        """Verify differential privacy guarantees for a noise mechanism.

        Args:
            epsilon: Privacy budget epsilon.
            delta: Failure probability delta.
            mechanism: Noise mechanism: 'gaussian', 'laplace', or 'exponential'.
            sensitivity: Query sensitivity.
            noise_scale: Noise scale parameter.

        Returns:
            DP verification result with guarantee_satisfied, privacy_level.
        """
        ...

    async def generate_privacy_report(
        self,
        membership_result: dict | None,
        attribute_results: list[dict],
        dp_result: dict | None,
        model_name: str,
    ) -> dict:
        """Aggregate privacy attack results into a comprehensive report.

        Args:
            membership_result: Membership inference result, or None.
            attribute_results: Attribute inference results.
            dp_result: DP verification result, or None.
            model_name: Human-readable model identifier.

        Returns:
            Privacy report with overall_privacy_risk and per-attack breakdown.
        """
        ...


@runtime_checkable
class ICoverageAnalyzer(Protocol):
    """Interface for code and input-space coverage analysis."""

    async def collect_code_coverage(
        self,
        source_paths: list[str],
        test_command: str,
        working_directory: str,
        include_branches: bool,
    ) -> dict:
        """Run tests with coverage instrumentation and collect coverage results.

        Args:
            source_paths: Source directory or file paths to instrument.
            test_command: Shell command to execute the test suite.
            working_directory: Directory to run the test command from.
            include_branches: Whether to collect branch coverage data.

        Returns:
            Coverage result dict with line_coverage, branch_coverage,
            per_file_coverage, files_analyzed.
        """
        ...

    async def enforce_threshold(
        self,
        coverage_data: dict,
        threshold: float,
        fail_on_decrease: bool,
        previous_coverage: float | None,
    ) -> dict:
        """Check whether coverage meets the configured threshold for CI/CD gating.

        Args:
            coverage_data: Coverage result from collect_code_coverage.
            threshold: Minimum required coverage fraction.
            fail_on_decrease: If True, also fail when coverage decreases.
            previous_coverage: Previous run coverage for regression comparison.

        Returns:
            Gate result dict with passed, reasons, recommended_exit_code.
        """
        ...

    async def generate_coverage_report(
        self,
        coverage_data: dict,
        output_format: str,
        output_path: str,
    ) -> dict:
        """Generate a coverage report in JSON or HTML format.

        Args:
            coverage_data: Coverage result dict to render.
            output_format: 'json' or 'html'.
            output_path: Destination file path.

        Returns:
            Report generation result with output_path, format, size_bytes.
        """
        ...


@runtime_checkable
class IPerformanceBenchmarker(Protocol):
    """Interface for latency, throughput, and resource profiling."""

    async def measure_latency(
        self,
        endpoint: str,
        sample_payload: dict,
        num_requests: int,
        concurrency: int,
        warmup_requests: int,
    ) -> dict:
        """Measure inference latency percentiles (P50, P90, P95, P99).

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload for each probe.
            num_requests: Total number of timed requests.
            concurrency: Number of simultaneous inflight requests.
            warmup_requests: Number of warmup requests before measurement.

        Returns:
            Latency result dict with p50_ms, p90_ms, p95_ms, p99_ms, error_rate.
        """
        ...

    async def measure_throughput(
        self,
        endpoint: str,
        sample_payload: dict,
        duration_seconds: int,
        concurrency: int,
    ) -> dict:
        """Measure sustained throughput (requests/sec) over a fixed duration.

        Args:
            endpoint: Model inference endpoint URL.
            sample_payload: Request payload for each probe.
            duration_seconds: Duration of the sustained load phase.
            concurrency: Number of simultaneous inflight requests.

        Returns:
            Throughput result dict with requests_per_second, error_rate.
        """
        ...

    async def compare_to_baseline(
        self,
        current_metrics: dict,
        baseline_metrics: dict,
        tolerance_percent: float,
    ) -> dict:
        """Compare current benchmark metrics against a stored baseline.

        Args:
            current_metrics: Current run metrics dict.
            baseline_metrics: Previous release baseline metrics dict.
            tolerance_percent: Allowed degradation percentage before regression.

        Returns:
            Comparison result with regressions, improvements, overall_status.
        """
        ...


@runtime_checkable
class IRegressionDetector(Protocol):
    """Interface for CI/CD quality gate regression detection."""

    async def compare_to_baseline(
        self,
        current_metrics: dict,
        baseline_metrics: dict,
        metric_configs: dict | None,
    ) -> dict:
        """Compare current evaluation metrics against a stored baseline.

        Args:
            current_metrics: Dict mapping metric_name -> current value.
            baseline_metrics: Dict mapping metric_name -> baseline value.
            metric_configs: Optional per-metric tolerance and direction config.

        Returns:
            Comparison result with regressions, improvements, overall_status.
        """
        ...

    async def evaluate_quality_gate(
        self,
        comparison_result: dict,
        statistical_result: dict | None,
        strict_mode: bool,
    ) -> dict:
        """Evaluate regression findings and produce a CI/CD gate decision.

        Args:
            comparison_result: Output from compare_to_baseline.
            statistical_result: Optional statistical significance test result.
            strict_mode: If True, apply stricter regression criteria.

        Returns:
            Gate decision dict with passed, exit_code, reasons, severity.
        """
        ...

    async def update_baseline(
        self,
        new_metrics: dict,
        baseline_file_path: str,
        approved_by: str,
        model_version: str,
    ) -> dict:
        """Update the stored baseline after an approved release.

        Args:
            new_metrics: New baseline metric values.
            baseline_file_path: Path to the baseline JSON file.
            approved_by: Approver identifier for audit trail.
            model_version: Model version being promoted to baseline.

        Returns:
            Baseline update result with file_path, updated_at, metric_count.
        """
        ...


@runtime_checkable
class ITestReportGenerator(Protocol):
    """Interface for automated multi-format test report generation."""

    async def aggregate_results(
        self,
        run_id: str,
        llm_results: list[dict] | None,
        rag_results: list[dict] | None,
        agent_results: list[dict] | None,
        red_team_results: list[dict] | None,
        adversarial_results: list[dict] | None,
        privacy_results: list[dict] | None,
        coverage_result: dict | None,
        performance_result: dict | None,
        regression_result: dict | None,
    ) -> dict:
        """Aggregate all test type results into a unified report payload.

        Args:
            run_id: Test run UUID for correlation.
            llm_results: LLM evaluation metric results.
            rag_results: RAG evaluation metric results.
            agent_results: Agent evaluation metric results.
            red_team_results: Red-team probe results.
            adversarial_results: Adversarial robustness results.
            privacy_results: Privacy attack results.
            coverage_result: Code coverage result.
            performance_result: Performance benchmark result.
            regression_result: Regression gate result.

        Returns:
            Aggregated report payload with executive summary and per-type details.
        """
        ...

    async def generate_report(
        self,
        aggregated_report: dict,
        output_format: str,
        output_path: str,
    ) -> dict:
        """Render and write a report file in the specified format.

        Args:
            aggregated_report: Full aggregated report payload.
            output_format: 'json' or 'pdf'.
            output_path: Destination file path.

        Returns:
            Report generation result with output_path, format, size_bytes.
        """
        ...

    async def generate_badge(
        self,
        aggregated_report: dict,
        badge_type: str,
    ) -> dict:
        """Generate a Shields.io badge URL for the test run result.

        Args:
            aggregated_report: Aggregated report payload.
            badge_type: Badge type: 'status', 'score', or 'coverage'.

        Returns:
            Badge dict with url, label, message, color, markdown.
        """
        ...

    async def distribute_via_webhook(
        self,
        aggregated_report: dict,
        webhook_url: str,
        headers: dict | None,
    ) -> dict:
        """POST the aggregated report to a webhook endpoint.

        Args:
            aggregated_report: Report payload to deliver.
            webhook_url: Destination webhook URL.
            headers: Optional additional HTTP headers.

        Returns:
            Delivery result with status_code, success, response_ms.
        """
        ...


@runtime_checkable
class ISyntheticDataTester(Protocol):
    """Interface for synthetic data fidelity and privacy validation."""

    async def compare_statistical_similarity(
        self,
        real_data: list[dict],
        synthetic_data: list[dict],
        numeric_columns: list[str],
        categorical_columns: list[str],
        threshold: float,
    ) -> dict:
        """Compute statistical similarity between real and synthetic datasets.

        Args:
            real_data: Reference real dataset as list of record dicts.
            synthetic_data: Synthetic dataset to evaluate.
            numeric_columns: Column names to treat as numeric.
            categorical_columns: Column names to treat as categorical.
            threshold: Minimum mean similarity score to pass.

        Returns:
            Statistical similarity result with mean_similarity, per_column_results, passed.
        """
        ...

    async def run_ml_utility_test(
        self,
        real_data: list[dict],
        synthetic_data: list[dict],
        feature_columns: list[str],
        target_column: str,
        model_type: str,
    ) -> dict:
        """Run the Train-on-Synthetic, Test-on-Real (TSTR) utility benchmark.

        Args:
            real_data: Real reference dataset.
            synthetic_data: Synthetic training dataset.
            feature_columns: Feature column names.
            target_column: Target classification column name.
            model_type: Classifier type: 'random_forest', 'logistic', or 'gradient_boosting'.

        Returns:
            ML utility result with tstr_accuracy, trtr_accuracy, utility_ratio, passed.
        """
        ...

    async def compute_fidelity_score(
        self,
        statistical_result: dict,
        ml_utility_result: dict,
        privacy_result: dict,
        column_dist_result: dict,
        threshold: float,
    ) -> dict:
        """Compute a weighted composite fidelity score from all quality dimensions.

        Args:
            statistical_result: Statistical similarity result.
            ml_utility_result: ML utility benchmark result.
            privacy_result: Privacy risk assessment result.
            column_dist_result: Column distribution comparison result.
            threshold: Minimum composite score to pass.

        Returns:
            Fidelity score dict with composite_score, per_dimension_scores, passed.
        """
        ...

    async def generate_validation_report(
        self,
        dataset_name: str,
        statistical_result: dict,
        ml_utility_result: dict,
        privacy_result: dict,
        column_dist_result: dict,
        fidelity_score_result: dict,
        fidelity_validator_result: dict | None,
    ) -> dict:
        """Assemble the full synthetic data validation report.

        Args:
            dataset_name: Human-readable dataset identifier.
            statistical_result: Statistical similarity result.
            ml_utility_result: TSTR benchmark result.
            privacy_result: Privacy risk result.
            column_dist_result: Per-column distribution comparison.
            fidelity_score_result: Composite fidelity score result.
            fidelity_validator_result: External fidelity validator result, or None.

        Returns:
            Validation report dict with executive summary and per-dimension details.
        """
        ...


__all__ = [
    "ITestSuiteRepository",
    "ITestRunRepository",
    "ITestResultRepository",
    "IRedTeamReportRepository",
    "ILLMEvaluator",
    "IRAGEvaluator",
    "IAgentEvaluator",
    "IRedTeamRunner",
    "IAdversarialTester",
    "IPrivacyTester",
    "ICoverageAnalyzer",
    "IPerformanceBenchmarker",
    "IRegressionDetector",
    "ITestReportGenerator",
    "ISyntheticDataTester",
]
