"""Shared fixtures for aumos-testing-harness tests.

Imports base fixtures from aumos_common.testing and adds
testing-harness-specific overrides.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_testing_harness.core.models import RunStatus, SuiteType, TestRun, TestSuite
from aumos_testing_harness.settings import Settings


@pytest.fixture()
def settings() -> Settings:
    """Provide test settings with safe defaults.

    Returns:
        Settings instance configured for testing.
    """
    return Settings(
        openai_api_key="sk-test-key",
        openai_model="gpt-4o",
        default_pass_threshold=0.7,
        max_eval_workers=2,
        eval_timeout_seconds=30,
        garak_enabled=False,
        giskard_enabled=False,
        red_team_max_attempts=5,
    )


@pytest.fixture()
def tenant_id() -> uuid.UUID:
    """Provide a stable test tenant UUID.

    Returns:
        Fixed tenant UUID for test isolation.
    """
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture()
def mock_tenant(tenant_id: uuid.UUID) -> MagicMock:
    """Provide a mock TenantContext.

    Args:
        tenant_id: The test tenant UUID.

    Returns:
        Mock TenantContext with tenant_id set.
    """
    tenant = MagicMock()
    tenant.tenant_id = tenant_id
    return tenant


@pytest.fixture()
def sample_suite_id() -> uuid.UUID:
    """Provide a stable test suite UUID.

    Returns:
        Fixed suite UUID.
    """
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture()
def sample_run_id() -> uuid.UUID:
    """Provide a stable test run UUID.

    Returns:
        Fixed run UUID.
    """
    return uuid.UUID("00000000-0000-0000-0000-000000000003")


@pytest.fixture()
def sample_test_suite(tenant_id: uuid.UUID, sample_suite_id: uuid.UUID) -> TestSuite:
    """Provide a sample TestSuite ORM object.

    Args:
        tenant_id: Test tenant UUID.
        sample_suite_id: Test suite UUID.

    Returns:
        TestSuite with llm configuration.
    """
    suite = TestSuite(
        id=str(sample_suite_id),
        tenant_id=str(tenant_id),
        name="Test LLM Suite",
        suite_type=SuiteType.LLM,
        config={
            "metrics": ["accuracy", "coherence"],
            "threshold": 0.7,
            "test_cases": [
                {
                    "input": "What is 2+2?",
                    "actual_output": "4",
                    "expected_output": "4",
                }
            ],
        },
        is_active=True,
    )
    return suite


@pytest.fixture()
def sample_test_run(tenant_id: uuid.UUID, sample_run_id: uuid.UUID, sample_suite_id: uuid.UUID) -> TestRun:
    """Provide a sample TestRun ORM object.

    Args:
        tenant_id: Test tenant UUID.
        sample_run_id: Test run UUID.
        sample_suite_id: Parent suite UUID.

    Returns:
        TestRun in PENDING status.
    """
    run = TestRun(
        id=str(sample_run_id),
        tenant_id=str(tenant_id),
        suite_id=str(sample_suite_id),
        status=RunStatus.PENDING,
        summary={},
    )
    return run


@pytest.fixture()
def mock_suite_repo(sample_test_suite: TestSuite) -> AsyncMock:
    """Provide a mock ITestSuiteRepository.

    Args:
        sample_test_suite: Default suite returned by get_by_id.

    Returns:
        AsyncMock implementing ITestSuiteRepository.
    """
    repo = AsyncMock()
    repo.get_by_id.return_value = sample_test_suite
    repo.list_all.return_value = MagicMock(items=[sample_test_suite], total=1, page=1, page_size=20)
    repo.create.return_value = sample_test_suite
    return repo


@pytest.fixture()
def mock_run_repo(sample_test_run: TestRun) -> AsyncMock:
    """Provide a mock ITestRunRepository.

    Args:
        sample_test_run: Default run returned by create/get_by_id.

    Returns:
        AsyncMock implementing ITestRunRepository.
    """
    repo = AsyncMock()
    repo.create.return_value = sample_test_run
    repo.get_by_id.return_value = sample_test_run
    repo.update_status.return_value = sample_test_run
    repo.list_all.return_value = MagicMock(items=[sample_test_run], total=1, page=1, page_size=20)
    return repo


@pytest.fixture()
def mock_result_repo() -> AsyncMock:
    """Provide a mock ITestResultRepository.

    Returns:
        AsyncMock implementing ITestResultRepository.
    """
    repo = AsyncMock()
    repo.bulk_create.return_value = []
    repo.list_by_run.return_value = MagicMock(items=[], total=0, page=1, page_size=50)
    return repo


@pytest.fixture()
def mock_report_repo() -> AsyncMock:
    """Provide a mock IRedTeamReportRepository.

    Returns:
        AsyncMock implementing IRedTeamReportRepository.
    """
    repo = AsyncMock()
    repo.get_by_run.return_value = []
    repo.create.return_value = MagicMock()
    return repo


@pytest.fixture()
def mock_llm_evaluator() -> AsyncMock:
    """Provide a mock ILLMEvaluator that returns passing scores.

    Returns:
        AsyncMock returning a list with one passing result.
    """
    evaluator = AsyncMock()
    evaluator.evaluate.return_value = [
        {
            "metric_name": "accuracy",
            "score": 0.85,
            "threshold": 0.7,
            "passed": True,
            "details": {"test_case_idx": 0, "input": "What is 2+2?"},
        }
    ]
    return evaluator


@pytest.fixture()
def mock_rag_evaluator() -> AsyncMock:
    """Provide a mock IRAGEvaluator that returns passing RAGAS scores.

    Returns:
        AsyncMock returning a list with RAGAS metric results.
    """
    evaluator = AsyncMock()
    evaluator.evaluate.return_value = [
        {
            "metric_name": "ragas_faithfulness",
            "score": 0.9,
            "threshold": 0.7,
            "passed": True,
            "details": {"question_idx": 0},
        },
        {
            "metric_name": "ragas_answer_relevancy",
            "score": 0.88,
            "threshold": 0.7,
            "passed": True,
            "details": {"question_idx": 0},
        },
    ]
    return evaluator


@pytest.fixture()
def mock_agent_evaluator() -> AsyncMock:
    """Provide a mock IAgentEvaluator that returns passing agent scores.

    Returns:
        AsyncMock returning agent metric results.
    """
    evaluator = AsyncMock()
    evaluator.evaluate.return_value = [
        {
            "metric_name": "agent_task_completion_rate",
            "score": 1.0,
            "threshold": 0.7,
            "passed": True,
            "details": {"task_idx": 0},
        }
    ]
    return evaluator


@pytest.fixture()
def mock_red_team_runner() -> AsyncMock:
    """Provide a mock IRedTeamRunner that returns no vulnerabilities.

    Returns:
        AsyncMock returning clean probe results.
    """
    runner = AsyncMock()
    runner.run_probes.return_value = [
        {
            "attack_type": "LLM01_prompt_injection",
            "success_rate": 0.0,
            "total_probes": 5,
            "successful_attacks": 0,
            "vulnerabilities": {"items": [], "tool": "mock"},
            "severity": "none",
        }
    ]
    return runner
