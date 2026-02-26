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


__all__ = [
    "ITestSuiteRepository",
    "ITestRunRepository",
    "ITestResultRepository",
    "IRedTeamReportRepository",
    "ILLMEvaluator",
    "IRAGEvaluator",
    "IAgentEvaluator",
    "IRedTeamRunner",
]
