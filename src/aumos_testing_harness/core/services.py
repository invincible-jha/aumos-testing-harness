"""Business logic services for aumos-testing-harness.

Services orchestrate evaluation runs, persist results, and publish Kafka events.
They are framework-agnostic: no FastAPI, no direct SQLAlchemy session access.

Service classes:
  - LLMEvalService   — orchestrates deepeval LLM metric evaluation
  - RAGEvalService   — orchestrates RAGAS RAG pipeline evaluation
  - AgentEvalService — orchestrates agent task capability evaluation
  - RedTeamService   — orchestrates OWASP LLM Top 10 red-team probes
"""

import uuid
from datetime import UTC, datetime

from aumos_common.auth import TenantContext
from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger
from aumos_common.pagination import PageRequest, PageResponse

from aumos_testing_harness.core.interfaces import (
    IAgentEvaluator,
    ILLMEvaluator,
    IRAGEvaluator,
    IRedTeamReportRepository,
    IRedTeamRunner,
    ITestResultRepository,
    ITestRunRepository,
    ITestSuiteRepository,
)
from aumos_testing_harness.core.models import (
    RedTeamAttackType,
    RedTeamReport,
    RunStatus,
    TestResult,
    TestRun,
    TestSuite,
)

logger = get_logger(__name__)


class LLMEvalService:
    """Orchestrates LLM quality evaluation using 14+ deepeval metrics.

    Manages the full lifecycle: creating a run, dispatching metric evaluations,
    persisting results, and publishing a completion event.

    Args:
        suite_repo: Repository for loading TestSuite definitions.
        run_repo: Repository for creating and updating TestRun records.
        result_repo: Repository for persisting per-metric TestResult rows.
        evaluator: LLM evaluator backend (deepeval implementation).
        publisher: Kafka event publisher for test lifecycle events.
    """

    def __init__(
        self,
        suite_repo: ITestSuiteRepository,
        run_repo: ITestRunRepository,
        result_repo: ITestResultRepository,
        evaluator: ILLMEvaluator,
        publisher: object,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            suite_repo: TestSuite data access layer.
            run_repo: TestRun data access layer.
            result_repo: TestResult data access layer.
            evaluator: LLM evaluation backend.
            publisher: Kafka event publisher.
        """
        self._suite_repo = suite_repo
        self._run_repo = run_repo
        self._result_repo = result_repo
        self._evaluator = evaluator
        self._publisher = publisher

    async def create_suite(
        self,
        tenant: TenantContext,
        name: str,
        config: dict,
        description: str | None = None,
    ) -> TestSuite:
        """Create a new LLM evaluation test suite.

        Args:
            tenant: Authenticated tenant context.
            name: Human-readable suite name.
            config: Suite config with metrics list, endpoint, test cases, threshold.
            description: Optional description.

        Returns:
            The persisted TestSuite.
        """
        suite = await self._suite_repo.create(
            tenant=tenant,
            name=name,
            suite_type="llm",
            config=config,
            description=description,
        )
        logger.info("LLM test suite created", suite_id=str(suite.id), tenant_id=str(tenant.tenant_id))
        # TODO: publish TestSuiteCreated event
        return suite

    async def list_suites(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestSuite]:
        """List LLM evaluation suites for a tenant.

        Args:
            tenant: Authenticated tenant context.
            page: Pagination parameters.

        Returns:
            Paginated list of TestSuite records.
        """
        return await self._suite_repo.list_all(tenant, page)

    async def get_suite(self, tenant: TenantContext, suite_id: uuid.UUID) -> TestSuite:
        """Retrieve a specific test suite.

        Args:
            tenant: Authenticated tenant context.
            suite_id: The UUID of the suite to retrieve.

        Returns:
            The matching TestSuite.

        Raises:
            NotFoundError: If the suite does not exist for this tenant.
        """
        suite = await self._suite_repo.get_by_id(suite_id, tenant)
        if suite is None:
            raise NotFoundError(resource="TestSuite", resource_id=str(suite_id))
        return suite

    async def run_suite(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        ci_build_id: str | None = None,
    ) -> TestRun:
        """Execute a test suite and persist all evaluation results.

        Creates a TestRun in PENDING status, dispatches metric evaluations,
        persists results, updates run status to COMPLETED or FAILED, and
        publishes a Kafka lifecycle event.

        Args:
            tenant: Authenticated tenant context.
            suite_id: The suite to execute.
            ci_build_id: Optional CI build identifier for correlation.

        Returns:
            The completed (or failed) TestRun with summary populated.

        Raises:
            NotFoundError: If the suite does not exist for this tenant.
        """
        suite = await self.get_suite(tenant, suite_id)

        run = await self._run_repo.create(tenant, suite_id, ci_build_id)
        logger.info(
            "LLM evaluation run started",
            run_id=str(run.id),
            suite_id=str(suite_id),
            tenant_id=str(tenant.tenant_id),
        )

        await self._run_repo.update_status(
            run.id,  # type: ignore[arg-type]
            tenant,
            RunStatus.RUNNING,
        )
        # TODO: publish TestRunStarted event

        try:
            metrics = suite.config.get("metrics", [])
            test_cases = suite.config.get("test_cases", [])
            threshold = suite.config.get("threshold", 0.7)

            raw_results = await self._evaluator.evaluate(
                metric_names=metrics,
                test_cases=test_cases,
                threshold=threshold,
            )

            await self._result_repo.bulk_create(
                tenant=tenant,
                run_id=run.id,  # type: ignore[arg-type]
                results=raw_results,
            )

            total = len(raw_results)
            passed = sum(1 for r in raw_results if r.get("passed", False))
            aggregate_score = sum(r.get("score", 0.0) for r in raw_results) / max(total, 1)

            summary = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "aggregate_score": round(aggregate_score, 4),
                "pass_rate": round(passed / max(total, 1), 4),
                "completed_at": datetime.now(UTC).isoformat(),
            }

            completed_run = await self._run_repo.update_status(
                run.id,  # type: ignore[arg-type]
                tenant,
                RunStatus.COMPLETED,
                summary,
            )
            logger.info(
                "LLM evaluation run completed",
                run_id=str(run.id),
                passed=passed,
                total=total,
                aggregate_score=aggregate_score,
            )
            # TODO: publish TestRunCompleted event
            return completed_run

        except Exception as exc:
            logger.error("LLM evaluation run failed", run_id=str(run.id), error=str(exc))
            failed_run = await self._run_repo.update_status(
                run.id,  # type: ignore[arg-type]
                tenant,
                RunStatus.FAILED,
                {"error": str(exc), "failed_at": datetime.now(UTC).isoformat()},
            )
            # TODO: publish TestRunFailed event
            return failed_run

    async def list_runs(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestRun]:
        """List all test runs for a tenant.

        Args:
            tenant: Authenticated tenant context.
            page: Pagination parameters.

        Returns:
            Paginated list of TestRun records.
        """
        return await self._run_repo.list_all(tenant, page)

    async def get_results(
        self,
        tenant: TenantContext,
        run_id: uuid.UUID,
        page: PageRequest,
    ) -> PageResponse[TestResult]:
        """Retrieve per-metric results for a completed test run.

        Args:
            tenant: Authenticated tenant context.
            run_id: The UUID of the test run.
            page: Pagination parameters.

        Returns:
            Paginated list of TestResult records.

        Raises:
            NotFoundError: If the run does not exist for this tenant.
        """
        run = await self._run_repo.get_by_id(run_id, tenant)
        if run is None:
            raise NotFoundError(resource="TestRun", resource_id=str(run_id))
        return await self._result_repo.list_by_run(run_id, tenant, page)


class RAGEvalService:
    """Orchestrates RAG pipeline evaluation using RAGAS metrics.

    Evaluates faithfulness, answer relevancy, context precision/recall,
    and answer correctness against a configured RAG endpoint.

    Args:
        suite_repo: Repository for loading TestSuite definitions.
        run_repo: Repository for creating and updating TestRun records.
        result_repo: Repository for persisting per-metric TestResult rows.
        evaluator: RAGAS evaluation backend.
        publisher: Kafka event publisher.
    """

    def __init__(
        self,
        suite_repo: ITestSuiteRepository,
        run_repo: ITestRunRepository,
        result_repo: ITestResultRepository,
        evaluator: IRAGEvaluator,
        publisher: object,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            suite_repo: TestSuite data access layer.
            run_repo: TestRun data access layer.
            result_repo: TestResult data access layer.
            evaluator: RAGAS evaluation backend.
            publisher: Kafka event publisher.
        """
        self._suite_repo = suite_repo
        self._run_repo = run_repo
        self._result_repo = result_repo
        self._evaluator = evaluator
        self._publisher = publisher

    async def run_suite(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        ci_build_id: str | None = None,
    ) -> TestRun:
        """Execute a RAG evaluation suite.

        Extracts question/answer/context data from suite config, runs RAGAS
        evaluation, and persists results.

        Args:
            tenant: Authenticated tenant context.
            suite_id: The suite to execute.
            ci_build_id: Optional CI build identifier.

        Returns:
            The completed TestRun.

        Raises:
            NotFoundError: If the suite does not exist for this tenant.
        """
        suite = await self._suite_repo.get_by_id(suite_id, tenant)
        if suite is None:
            raise NotFoundError(resource="TestSuite", resource_id=str(suite_id))

        run = await self._run_repo.create(tenant, suite_id, ci_build_id)
        await self._run_repo.update_status(run.id, tenant, RunStatus.RUNNING)  # type: ignore[arg-type]

        logger.info(
            "RAG evaluation run started",
            run_id=str(run.id),
            suite_id=str(suite_id),
            tenant_id=str(tenant.tenant_id),
        )

        try:
            test_cases = suite.config.get("test_cases", [])
            questions = [tc["question"] for tc in test_cases]
            answers = [tc["answer"] for tc in test_cases]
            contexts = [tc.get("contexts", []) for tc in test_cases]
            ground_truths = [tc.get("ground_truth") for tc in test_cases]

            raw_results = await self._evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths if any(gt is not None for gt in ground_truths) else None,
            )

            await self._result_repo.bulk_create(
                tenant=tenant,
                run_id=run.id,  # type: ignore[arg-type]
                results=raw_results,
            )

            total = len(raw_results)
            passed = sum(1 for r in raw_results if r.get("passed", False))
            aggregate_score = sum(r.get("score", 0.0) for r in raw_results) / max(total, 1)

            summary = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "aggregate_score": round(aggregate_score, 4),
                "pass_rate": round(passed / max(total, 1), 4),
                "completed_at": datetime.now(UTC).isoformat(),
            }

            completed_run = await self._run_repo.update_status(
                run.id, tenant, RunStatus.COMPLETED, summary  # type: ignore[arg-type]
            )
            logger.info(
                "RAG evaluation run completed",
                run_id=str(run.id),
                passed=passed,
                total=total,
                aggregate_score=aggregate_score,
            )
            return completed_run

        except Exception as exc:
            logger.error("RAG evaluation run failed", run_id=str(run.id), error=str(exc))
            return await self._run_repo.update_status(
                run.id,  # type: ignore[arg-type]
                tenant,
                RunStatus.FAILED,
                {"error": str(exc), "failed_at": datetime.now(UTC).isoformat()},
            )


class AgentEvalService:
    """Orchestrates multi-step agent capability evaluation.

    Tests task completion rate, tool usage accuracy, multi-step reasoning,
    and efficiency against recorded or live agent trajectories.

    Args:
        suite_repo: Repository for loading TestSuite definitions.
        run_repo: Repository for creating and updating TestRun records.
        result_repo: Repository for persisting per-metric TestResult rows.
        evaluator: Agent evaluation backend.
        publisher: Kafka event publisher.
    """

    def __init__(
        self,
        suite_repo: ITestSuiteRepository,
        run_repo: ITestRunRepository,
        result_repo: ITestResultRepository,
        evaluator: IAgentEvaluator,
        publisher: object,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            suite_repo: TestSuite data access layer.
            run_repo: TestRun data access layer.
            result_repo: TestResult data access layer.
            evaluator: Agent evaluation backend.
            publisher: Kafka event publisher.
        """
        self._suite_repo = suite_repo
        self._run_repo = run_repo
        self._result_repo = result_repo
        self._evaluator = evaluator
        self._publisher = publisher

    async def run_suite(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        ci_build_id: str | None = None,
    ) -> TestRun:
        """Execute an agent evaluation suite.

        Extracts task definitions and pre-recorded agent trajectories from suite
        config, dispatches evaluation, and persists scored results.

        Args:
            tenant: Authenticated tenant context.
            suite_id: The suite to execute.
            ci_build_id: Optional CI build identifier.

        Returns:
            The completed TestRun.

        Raises:
            NotFoundError: If the suite does not exist for this tenant.
        """
        suite = await self._suite_repo.get_by_id(suite_id, tenant)
        if suite is None:
            raise NotFoundError(resource="TestSuite", resource_id=str(suite_id))

        run = await self._run_repo.create(tenant, suite_id, ci_build_id)
        await self._run_repo.update_status(run.id, tenant, RunStatus.RUNNING)  # type: ignore[arg-type]

        logger.info(
            "Agent evaluation run started",
            run_id=str(run.id),
            suite_id=str(suite_id),
            tenant_id=str(tenant.tenant_id),
        )

        try:
            task_definitions = suite.config.get("task_definitions", [])
            agent_trajectories = suite.config.get("agent_trajectories", [])
            threshold = suite.config.get("threshold", 0.7)

            raw_results = await self._evaluator.evaluate(
                task_definitions=task_definitions,
                agent_trajectories=agent_trajectories,
                threshold=threshold,
            )

            await self._result_repo.bulk_create(
                tenant=tenant,
                run_id=run.id,  # type: ignore[arg-type]
                results=raw_results,
            )

            total = len(raw_results)
            passed = sum(1 for r in raw_results if r.get("passed", False))
            aggregate_score = sum(r.get("score", 0.0) for r in raw_results) / max(total, 1)

            summary = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "aggregate_score": round(aggregate_score, 4),
                "pass_rate": round(passed / max(total, 1), 4),
                "completed_at": datetime.now(UTC).isoformat(),
            }

            completed_run = await self._run_repo.update_status(
                run.id, tenant, RunStatus.COMPLETED, summary  # type: ignore[arg-type]
            )
            logger.info(
                "Agent evaluation run completed",
                run_id=str(run.id),
                passed=passed,
                total=total,
            )
            return completed_run

        except Exception as exc:
            logger.error("Agent evaluation run failed", run_id=str(run.id), error=str(exc))
            return await self._run_repo.update_status(
                run.id,  # type: ignore[arg-type]
                tenant,
                RunStatus.FAILED,
                {"error": str(exc), "failed_at": datetime.now(UTC).isoformat()},
            )


class RedTeamService:
    """Orchestrates OWASP LLM Top 10 red-team assessments.

    Launches Garak and Giskard probe runners against a target model endpoint,
    aggregates vulnerability findings, persists sanitised reports, and publishes
    events when critical vulnerabilities are detected.

    Args:
        run_repo: Repository for creating and updating TestRun records.
        report_repo: Repository for persisting RedTeamReport records.
        runner: Red-team probe runner backend.
        publisher: Kafka event publisher.
    """

    def __init__(
        self,
        run_repo: ITestRunRepository,
        report_repo: IRedTeamReportRepository,
        runner: IRedTeamRunner,
        publisher: object,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            run_repo: TestRun data access layer.
            report_repo: RedTeamReport data access layer.
            runner: Red-team probe runner.
            publisher: Kafka event publisher.
        """
        self._run_repo = run_repo
        self._report_repo = report_repo
        self._runner = runner
        self._publisher = publisher

    async def launch_assessment(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        target_endpoint: str,
        owasp_categories: list[str] | None = None,
        max_attempts_per_category: int = 50,
    ) -> TestRun:
        """Launch a full red-team assessment against a model endpoint.

        Creates a TestRun, runs OWASP probes across all requested categories,
        persists sanitised vulnerability reports, and updates run status.

        Args:
            tenant: Authenticated tenant context.
            suite_id: The associated test suite UUID.
            target_endpoint: URL of the model inference endpoint to probe.
            owasp_categories: List of OWASP category IDs to probe. Defaults to all 10.
            max_attempts_per_category: Maximum probe attempts per attack category.

        Returns:
            The completed TestRun with vulnerability summary.
        """
        attack_types = self._resolve_attack_types(owasp_categories)

        run = await self._run_repo.create(tenant, suite_id)
        await self._run_repo.update_status(run.id, tenant, RunStatus.RUNNING)  # type: ignore[arg-type]

        logger.info(
            "Red-team assessment started",
            run_id=str(run.id),
            target_endpoint=target_endpoint,
            categories=len(attack_types),
            tenant_id=str(tenant.tenant_id),
        )
        # TODO: publish RedTeamStarted event

        try:
            probe_results = await self._runner.run_probes(
                target_endpoint=target_endpoint,
                attack_types=attack_types,
                max_attempts=max_attempts_per_category,
            )

            total_vulnerabilities = 0
            critical_found = False

            for probe_result in probe_results:
                report = await self._report_repo.create(
                    tenant=tenant,
                    run_id=run.id,  # type: ignore[arg-type]
                    attack_type=RedTeamAttackType(probe_result["attack_type"]),
                    success_rate=probe_result["success_rate"],
                    vulnerabilities=probe_result.get("vulnerabilities", {}),
                    total_probes=probe_result.get("total_probes", 0),
                    successful_attacks=probe_result.get("successful_attacks", 0),
                )

                vuln_count = len(probe_result.get("vulnerabilities", {}).get("items", []))
                total_vulnerabilities += vuln_count

                if probe_result["success_rate"] > 0.5:
                    critical_found = True
                    logger.warning(
                        "High red-team success rate detected",
                        run_id=str(run.id),
                        attack_type=probe_result["attack_type"],
                        success_rate=probe_result["success_rate"],
                    )
                    # TODO: publish VulnerabilityDetected event immediately

                logger.info(
                    "Red-team probe completed",
                    run_id=str(run.id),
                    attack_type=probe_result["attack_type"],
                    success_rate=probe_result["success_rate"],
                    report_id=str(report.id),
                )

            summary = {
                "categories_tested": len(probe_results),
                "total_vulnerabilities": total_vulnerabilities,
                "critical_found": critical_found,
                "completed_at": datetime.now(UTC).isoformat(),
            }

            completed_run = await self._run_repo.update_status(
                run.id, tenant, RunStatus.COMPLETED, summary  # type: ignore[arg-type]
            )
            logger.info(
                "Red-team assessment completed",
                run_id=str(run.id),
                total_vulnerabilities=total_vulnerabilities,
                critical_found=critical_found,
            )
            # TODO: publish RedTeamCompleted event
            return completed_run

        except Exception as exc:
            logger.error("Red-team assessment failed", run_id=str(run.id), error=str(exc))
            return await self._run_repo.update_status(
                run.id,  # type: ignore[arg-type]
                tenant,
                RunStatus.FAILED,
                {"error": str(exc), "failed_at": datetime.now(UTC).isoformat()},
            )

    async def get_report(
        self,
        tenant: TenantContext,
        run_id: uuid.UUID,
    ) -> list[RedTeamReport]:
        """Retrieve all red-team reports for a completed assessment run.

        Args:
            tenant: Authenticated tenant context.
            run_id: The UUID of the test run.

        Returns:
            List of RedTeamReport records for the run.

        Raises:
            NotFoundError: If the run does not exist for this tenant.
        """
        run = await self._run_repo.get_by_id(run_id, tenant)
        if run is None:
            raise NotFoundError(resource="TestRun", resource_id=str(run_id))
        return await self._report_repo.get_by_run(run_id, tenant)

    def _resolve_attack_types(self, owasp_categories: list[str] | None) -> list[RedTeamAttackType]:
        """Resolve OWASP category strings to RedTeamAttackType enum values.

        If no categories are specified, returns all 10 OWASP LLM categories.

        Args:
            owasp_categories: Optional list of OWASP category IDs (e.g. ["LLM01", "LLM06"]).

        Returns:
            List of RedTeamAttackType values to probe.
        """
        owasp_map: dict[str, RedTeamAttackType] = {
            "LLM01": RedTeamAttackType.PROMPT_INJECTION,
            "LLM02": RedTeamAttackType.INSECURE_OUTPUT,
            "LLM03": RedTeamAttackType.TRAINING_DATA_POISONING,
            "LLM04": RedTeamAttackType.MODEL_DOS,
            "LLM05": RedTeamAttackType.SUPPLY_CHAIN,
            "LLM06": RedTeamAttackType.SENSITIVE_DISCLOSURE,
            "LLM07": RedTeamAttackType.INSECURE_PLUGIN,
            "LLM08": RedTeamAttackType.EXCESSIVE_AGENCY,
            "LLM09": RedTeamAttackType.OVERRELIANCE,
            "LLM10": RedTeamAttackType.MODEL_THEFT,
        }

        if not owasp_categories:
            return list(owasp_map.values())

        resolved: list[RedTeamAttackType] = []
        for category in owasp_categories:
            attack_type = owasp_map.get(category.upper())
            if attack_type:
                resolved.append(attack_type)
            else:
                logger.warning("Unknown OWASP category skipped", category=category)

        return resolved if resolved else list(owasp_map.values())


__all__ = [
    "LLMEvalService",
    "RAGEvalService",
    "AgentEvalService",
    "RedTeamService",
]
