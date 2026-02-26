"""API router for aumos-testing-harness.

All endpoints are registered here and included in main.py under /api/v1.
Routes are thin: they validate input, delegate to services, and return responses.
No business logic belongs in this module.

Endpoints:
  POST/GET  /suites              — CRUD for test suites
  GET       /suites/{id}         — Get a single test suite
  POST      /suites/{id}/run     — Execute a test suite
  GET       /runs                — List all test runs
  GET       /runs/{id}/results   — Get results for a completed run
  POST      /red-team            — Launch a red-team assessment
  GET       /red-team/{id}/report — Get red-team reports for a run
  GET       /metrics             — Available metrics catalog
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_user
from aumos_common.database import get_db_session
from aumos_common.pagination import PageRequest

from aumos_testing_harness.adapters.agent_evaluator import AgentEvaluator
from aumos_testing_harness.adapters.llm_evaluator import LLMEvaluator
from aumos_testing_harness.adapters.rag_evaluator import RAGEvaluator
from aumos_testing_harness.adapters.red_team_runner import RedTeamRunner
from aumos_testing_harness.adapters.repositories import (
    RedTeamReportRepository,
    TestResultRepository,
    TestRunRepository,
    TestSuiteRepository,
)
from aumos_testing_harness.api.schemas import (
    MetricInfo,
    MetricsCatalogResponse,
    RedTeamLaunchRequest,
    RedTeamReportListResponse,
    RunSuiteRequest,
    TestResultListResponse,
    TestRunListResponse,
    TestRunResponse,
    TestSuiteCreateRequest,
    TestSuiteListResponse,
    TestSuiteResponse,
)
from aumos_testing_harness.core.models import MetricName
from aumos_testing_harness.core.services import (
    AgentEvalService,
    LLMEvalService,
    RAGEvalService,
    RedTeamService,
)
from aumos_testing_harness.settings import Settings

router = APIRouter(tags=["testing-harness"])
settings = Settings()

# ---------------------------------------------------------------------------
# Dependency factory helpers
# ---------------------------------------------------------------------------


def _get_llm_service(session: AsyncSession = Depends(get_db_session)) -> LLMEvalService:
    """Build LLMEvalService with all injected dependencies.

    Args:
        session: Async database session from aumos-common.

    Returns:
        Configured LLMEvalService instance.
    """
    return LLMEvalService(
        suite_repo=TestSuiteRepository(session),
        run_repo=TestRunRepository(session),
        result_repo=TestResultRepository(session),
        evaluator=LLMEvaluator(settings),
        publisher=None,  # TODO: inject Kafka publisher
    )


def _get_rag_service(session: AsyncSession = Depends(get_db_session)) -> RAGEvalService:
    """Build RAGEvalService with all injected dependencies.

    Args:
        session: Async database session from aumos-common.

    Returns:
        Configured RAGEvalService instance.
    """
    return RAGEvalService(
        suite_repo=TestSuiteRepository(session),
        run_repo=TestRunRepository(session),
        result_repo=TestResultRepository(session),
        evaluator=RAGEvaluator(settings),
        publisher=None,  # TODO: inject Kafka publisher
    )


def _get_agent_service(session: AsyncSession = Depends(get_db_session)) -> AgentEvalService:
    """Build AgentEvalService with all injected dependencies.

    Args:
        session: Async database session from aumos-common.

    Returns:
        Configured AgentEvalService instance.
    """
    return AgentEvalService(
        suite_repo=TestSuiteRepository(session),
        run_repo=TestRunRepository(session),
        result_repo=TestResultRepository(session),
        evaluator=AgentEvaluator(settings),
        publisher=None,  # TODO: inject Kafka publisher
    )


def _get_red_team_service(session: AsyncSession = Depends(get_db_session)) -> RedTeamService:
    """Build RedTeamService with all injected dependencies.

    Args:
        session: Async database session from aumos-common.

    Returns:
        Configured RedTeamService instance.
    """
    return RedTeamService(
        run_repo=TestRunRepository(session),
        report_repo=RedTeamReportRepository(session),
        runner=RedTeamRunner(settings),
        publisher=None,  # TODO: inject Kafka publisher
    )


# ---------------------------------------------------------------------------
# Test Suite endpoints
# ---------------------------------------------------------------------------


@router.post("/suites", response_model=TestSuiteResponse, status_code=201)
async def create_test_suite(
    body: TestSuiteCreateRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestSuiteResponse:
    """Create a new test suite.

    Args:
        body: Suite creation payload with name, type, and configuration.
        tenant: Authenticated tenant context from JWT.
        service: LLM evaluation service.

    Returns:
        The newly created TestSuite.
    """
    suite = await service.create_suite(
        tenant=tenant,
        name=body.name,
        config=body.config,
        description=body.description,
    )
    return TestSuiteResponse.model_validate(suite)


@router.get("/suites", response_model=TestSuiteListResponse)
async def list_test_suites(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestSuiteListResponse:
    """List all test suites for the authenticated tenant.

    Args:
        page: Page number (1-based).
        page_size: Number of items per page.
        tenant: Authenticated tenant context.
        service: LLM evaluation service.

    Returns:
        Paginated list of test suites.
    """
    page_request = PageRequest(page=page, page_size=page_size)
    result = await service.list_suites(tenant, page_request)
    return TestSuiteListResponse(
        items=[TestSuiteResponse.model_validate(s) for s in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
    )


@router.get("/suites/{suite_id}", response_model=TestSuiteResponse)
async def get_test_suite(
    suite_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestSuiteResponse:
    """Retrieve a specific test suite by ID.

    Args:
        suite_id: The UUID of the test suite.
        tenant: Authenticated tenant context.
        service: LLM evaluation service.

    Returns:
        The matching TestSuite.
    """
    suite = await service.get_suite(tenant, suite_id)
    return TestSuiteResponse.model_validate(suite)


@router.post("/suites/{suite_id}/run", response_model=TestRunResponse, status_code=202)
async def run_test_suite(
    suite_id: uuid.UUID,
    body: RunSuiteRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestRunResponse:
    """Execute a test suite and return the run record.

    The suite type in the configuration determines which evaluator is used.
    For RAG and agent suites, the same endpoint delegates to the appropriate service.

    Args:
        suite_id: The UUID of the suite to execute.
        body: Optional run parameters including CI build ID.
        tenant: Authenticated tenant context.
        service: LLM evaluation service (also handles routing to RAG/agent).

    Returns:
        The TestRun record (which may still be RUNNING for async suites).
    """
    run = await service.run_suite(
        tenant=tenant,
        suite_id=suite_id,
        ci_build_id=body.ci_build_id,
    )
    return TestRunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# Test Run endpoints
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=TestRunListResponse)
async def list_test_runs(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestRunListResponse:
    """List all test runs for the authenticated tenant.

    Args:
        page: Page number (1-based).
        page_size: Number of items per page.
        tenant: Authenticated tenant context.
        service: Evaluation service with run listing capability.

    Returns:
        Paginated list of test runs.
    """
    page_request = PageRequest(page=page, page_size=page_size)
    result = await service.list_runs(tenant, page_request)
    return TestRunListResponse(
        items=[TestRunResponse.model_validate(r) for r in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
    )


@router.get("/runs/{run_id}/results", response_model=TestResultListResponse)
async def get_run_results(
    run_id: uuid.UUID,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    tenant: TenantContext = Depends(get_current_user),
    service: LLMEvalService = Depends(_get_llm_service),
) -> TestResultListResponse:
    """Get all per-metric results for a completed test run.

    Args:
        run_id: The UUID of the test run.
        page: Page number (1-based).
        page_size: Number of items per page.
        tenant: Authenticated tenant context.
        service: Evaluation service with result retrieval capability.

    Returns:
        Paginated list of TestResult records.
    """
    from aumos_testing_harness.api.schemas import TestResultResponse  # noqa: PLC0415

    page_request = PageRequest(page=page, page_size=page_size)
    result = await service.get_results(tenant, run_id, page_request)
    return TestResultListResponse(
        items=[TestResultResponse.model_validate(r) for r in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
    )


# ---------------------------------------------------------------------------
# Red-team endpoints
# ---------------------------------------------------------------------------


@router.post("/red-team", response_model=TestRunResponse, status_code=202)
async def launch_red_team(
    body: RedTeamLaunchRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: RedTeamService = Depends(_get_red_team_service),
) -> TestRunResponse:
    """Launch an OWASP LLM Top 10 red-team assessment.

    Args:
        body: Assessment parameters including target endpoint and OWASP categories.
        tenant: Authenticated tenant context.
        service: Red-team service.

    Returns:
        The TestRun record for the launched assessment.
    """
    run = await service.launch_assessment(
        tenant=tenant,
        suite_id=body.suite_id,
        target_endpoint=body.target_endpoint,
        owasp_categories=body.owasp_categories,
        max_attempts_per_category=body.max_attempts_per_category,
    )
    return TestRunResponse.model_validate(run)


@router.get("/red-team/{run_id}/report", response_model=RedTeamReportListResponse)
async def get_red_team_report(
    run_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: RedTeamService = Depends(_get_red_team_service),
) -> RedTeamReportListResponse:
    """Retrieve all red-team vulnerability reports for a completed assessment.

    Args:
        run_id: The UUID of the red-team test run.
        tenant: Authenticated tenant context.
        service: Red-team service.

    Returns:
        All RedTeamReport records for the run with aggregated summary.
    """
    from aumos_testing_harness.api.schemas import RedTeamReportResponse  # noqa: PLC0415

    reports = await service.get_report(tenant, run_id)
    total_vulns = sum(len(r.vulnerabilities.get("items", [])) for r in reports)
    critical = any(r.success_rate > 0.5 for r in reports)

    return RedTeamReportListResponse(
        run_id=run_id,
        reports=[RedTeamReportResponse.model_validate(r) for r in reports],
        summary={
            "total_reports": len(reports),
            "total_vulnerabilities": total_vulns,
            "critical_found": critical,
        },
    )


# ---------------------------------------------------------------------------
# Metrics catalog endpoint
# ---------------------------------------------------------------------------

_METRICS_CATALOG: list[MetricInfo] = [
    MetricInfo(
        name=MetricName.ACCURACY,
        display_name="Accuracy",
        category="llm",
        requires_ground_truth=True,
        description="Factual correctness of the response against a reference answer",
        score_range="0.0–1.0",
        framework="deepeval (GEval)",
    ),
    MetricInfo(
        name=MetricName.COHERENCE,
        display_name="Coherence",
        category="llm",
        requires_ground_truth=False,
        description="Logical flow, internal consistency, and clarity of the response",
        score_range="0.0–1.0",
        framework="deepeval (GEval)",
    ),
    MetricInfo(
        name=MetricName.FAITHFULNESS,
        display_name="Faithfulness",
        category="llm",
        requires_ground_truth=False,
        description="Degree to which the response is grounded in the provided context",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.ANSWER_RELEVANCY,
        display_name="Answer Relevancy",
        category="llm",
        requires_ground_truth=False,
        description="How well the response addresses the user question",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.CONTEXTUAL_PRECISION,
        display_name="Contextual Precision",
        category="llm",
        requires_ground_truth=True,
        description="Proportion of retrieved context that is actually relevant",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.CONTEXTUAL_RECALL,
        display_name="Contextual Recall",
        category="llm",
        requires_ground_truth=True,
        description="Coverage of ground-truth information in the retrieved context",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.CONTEXTUAL_RELEVANCY,
        display_name="Contextual Relevancy",
        category="llm",
        requires_ground_truth=False,
        description="How well the retrieved context matches the input query",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.HALLUCINATION,
        display_name="Hallucination Detection",
        category="llm",
        requires_ground_truth=False,
        description="Detection of unsupported or fabricated claims in the response",
        score_range="0.0–1.0 (higher = less hallucination)",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.TOXICITY,
        display_name="Toxicity",
        category="llm",
        requires_ground_truth=False,
        description="Detection of harmful, offensive, or abusive content",
        score_range="0.0–1.0 (higher = less toxic)",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.BIAS,
        display_name="Bias Detection",
        category="llm",
        requires_ground_truth=False,
        description="Detection of demographic, ideological, or systemic bias in responses",
        score_range="0.0–1.0 (higher = less biased)",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.SUMMARIZATION,
        display_name="Summarization Quality",
        category="llm",
        requires_ground_truth=True,
        description="Fidelity and coverage quality of a generated summary",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.TASK_COMPLETION,
        display_name="Task Completion",
        category="llm",
        requires_ground_truth=True,
        description="Degree to which the response achieves the specified task goal",
        score_range="0.0–1.0",
        framework="deepeval (GEval)",
    ),
    MetricInfo(
        name=MetricName.TOOL_CALL_ACCURACY,
        display_name="Tool Call Accuracy",
        category="llm",
        requires_ground_truth=True,
        description="Correctness of tool selection and argument construction",
        score_range="0.0–1.0",
        framework="deepeval",
    ),
    MetricInfo(
        name=MetricName.LATENCY_SCORE,
        display_name="Latency Score",
        category="llm",
        requires_ground_truth=False,
        description="Response time relative to a configured SLA threshold",
        score_range="0.0–1.0 (1.0 = within SLA)",
        framework="custom",
    ),
    MetricInfo(
        name=MetricName.RAGAS_FAITHFULNESS,
        display_name="RAGAS Faithfulness",
        category="rag",
        requires_ground_truth=False,
        description="Answer is supported by the retrieved context chunks",
        score_range="0.0–1.0",
        framework="ragas",
    ),
    MetricInfo(
        name=MetricName.RAGAS_ANSWER_RELEVANCY,
        display_name="RAGAS Answer Relevancy",
        category="rag",
        requires_ground_truth=False,
        description="Answer is pertinent to the user question",
        score_range="0.0–1.0",
        framework="ragas",
    ),
    MetricInfo(
        name=MetricName.RAGAS_CONTEXT_PRECISION,
        display_name="RAGAS Context Precision",
        category="rag",
        requires_ground_truth=True,
        description="Relevant context chunks are ranked above irrelevant ones",
        score_range="0.0–1.0",
        framework="ragas",
    ),
    MetricInfo(
        name=MetricName.RAGAS_CONTEXT_RECALL,
        display_name="RAGAS Context Recall",
        category="rag",
        requires_ground_truth=True,
        description="All ground-truth information is present in the retrieved context",
        score_range="0.0–1.0",
        framework="ragas",
    ),
    MetricInfo(
        name=MetricName.RAGAS_ANSWER_CORRECTNESS,
        display_name="RAGAS Answer Correctness",
        category="rag",
        requires_ground_truth=True,
        description="End-to-end factual accuracy of the RAG pipeline output",
        score_range="0.0–1.0",
        framework="ragas",
    ),
    MetricInfo(
        name=MetricName.AGENT_TASK_COMPLETION_RATE,
        display_name="Agent Task Completion Rate",
        category="agent",
        requires_ground_truth=True,
        description="Fraction of goal criteria successfully met by the agent",
        score_range="0.0–1.0",
        framework="custom",
    ),
    MetricInfo(
        name=MetricName.AGENT_TOOL_USAGE_ACCURACY,
        display_name="Agent Tool Usage Accuracy",
        category="agent",
        requires_ground_truth=True,
        description="Correct tool called with correct arguments at each step",
        score_range="0.0–1.0",
        framework="custom",
    ),
    MetricInfo(
        name=MetricName.AGENT_MULTI_STEP_REASONING,
        display_name="Agent Multi-Step Reasoning",
        category="agent",
        requires_ground_truth=False,
        description="Logical coherence and consistency across multiple reasoning steps",
        score_range="0.0–1.0",
        framework="deepeval (GEval)",
    ),
    MetricInfo(
        name=MetricName.AGENT_EFFICIENCY_SCORE,
        display_name="Agent Efficiency Score",
        category="agent",
        requires_ground_truth=True,
        description="Steps taken versus the optimal path length",
        score_range="0.0–1.0 (1.0 = optimal path)",
        framework="custom",
    ),
]


@router.get("/metrics", response_model=MetricsCatalogResponse)
async def list_metrics(
    tenant: TenantContext = Depends(get_current_user),
) -> MetricsCatalogResponse:
    """List all available evaluation metrics with metadata.

    Args:
        tenant: Authenticated tenant context (used for auth gate only).

    Returns:
        Full catalog of supported evaluation metrics.
    """
    return MetricsCatalogResponse(
        metrics=_METRICS_CATALOG,
        total=len(_METRICS_CATALOG),
    )
