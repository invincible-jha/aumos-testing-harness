"""Repository layer tests for aumos-testing-harness.

Tests the SQLAlchemy repository implementations using mocked sessions.
For full integration tests with real PostgreSQL, use testcontainers fixtures.

Coverage target: 60% for adapters/ modules.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_testing_harness.adapters.repositories import (
    RedTeamReportRepository,
    TestResultRepository,
    TestRunRepository,
    TestSuiteRepository,
)
from aumos_testing_harness.core.models import RedTeamAttackType, RunStatus, SuiteType


@pytest.fixture()
def mock_session() -> AsyncMock:
    """Provide a mock AsyncSession.

    Returns:
        AsyncMock with execute, add, commit, refresh methods.
    """
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture()
def mock_tenant() -> MagicMock:
    """Provide a mock TenantContext.

    Returns:
        Mock with tenant_id set.
    """
    tenant = MagicMock()
    tenant.tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    return tenant


class TestTestSuiteRepository:
    """Tests for TestSuiteRepository."""

    def test_init_sets_session(self, mock_session: AsyncMock) -> None:
        """Repository stores the injected session.

        Args:
            mock_session: Mock database session.
        """
        repo = TestSuiteRepository(mock_session)
        assert repo.session is mock_session

    @pytest.mark.asyncio()
    async def test_create_adds_and_commits(
        self, mock_session: AsyncMock, mock_tenant: MagicMock
    ) -> None:
        """create() adds a TestSuite to the session and commits.

        Args:
            mock_session: Mock database session.
            mock_tenant: Mock tenant context.
        """
        repo = TestSuiteRepository(mock_session)

        mock_suite = MagicMock()
        with patch("aumos_testing_harness.adapters.repositories.TestSuite", return_value=mock_suite):
            result = await repo.create(
                tenant=mock_tenant,
                name="My Suite",
                suite_type="llm",
                config={"metrics": ["accuracy"]},
            )

        mock_session.add.assert_called_once_with(mock_suite)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()


class TestTestRunRepository:
    """Tests for TestRunRepository."""

    @pytest.mark.asyncio()
    async def test_create_sets_pending_status(
        self, mock_session: AsyncMock, mock_tenant: MagicMock
    ) -> None:
        """create() sets the initial status to PENDING.

        Args:
            mock_session: Mock database session.
            mock_tenant: Mock tenant context.
        """
        repo = TestRunRepository(mock_session)

        mock_run = MagicMock()
        mock_run.status = RunStatus.PENDING

        with patch("aumos_testing_harness.adapters.repositories.TestRun", return_value=mock_run):
            result = await repo.create(
                tenant=mock_tenant,
                suite_id=uuid.uuid4(),
            )

        assert result.status == RunStatus.PENDING
        mock_session.add.assert_called_once_with(mock_run)
        mock_session.commit.assert_called_once()


class TestTestResultRepository:
    """Tests for TestResultRepository."""

    @pytest.mark.asyncio()
    async def test_bulk_create_adds_all_results(
        self, mock_session: AsyncMock, mock_tenant: MagicMock
    ) -> None:
        """bulk_create() adds one TestResult per entry and commits once.

        Args:
            mock_session: Mock database session.
            mock_tenant: Mock tenant context.
        """
        repo = TestResultRepository(mock_session)

        results_data = [
            {
                "metric_name": "accuracy",
                "score": 0.9,
                "threshold": 0.7,
                "passed": True,
                "details": {},
            },
            {
                "metric_name": "coherence",
                "score": 0.8,
                "threshold": 0.7,
                "passed": True,
                "details": {},
            },
        ]

        with patch("aumos_testing_harness.adapters.repositories.TestResult") as MockTestResult:
            MockTestResult.return_value = MagicMock()
            await repo.bulk_create(
                tenant=mock_tenant,
                run_id=uuid.uuid4(),
                results=results_data,
            )

        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()


class TestRedTeamReportRepository:
    """Tests for RedTeamReportRepository."""

    @pytest.mark.asyncio()
    async def test_create_persists_report(
        self, mock_session: AsyncMock, mock_tenant: MagicMock
    ) -> None:
        """create() persists a RedTeamReport with all required fields.

        Args:
            mock_session: Mock database session.
            mock_tenant: Mock tenant context.
        """
        repo = RedTeamReportRepository(mock_session)

        mock_report = MagicMock()
        with patch(
            "aumos_testing_harness.adapters.repositories.RedTeamReport", return_value=mock_report
        ):
            result = await repo.create(
                tenant=mock_tenant,
                run_id=uuid.uuid4(),
                attack_type=RedTeamAttackType.PROMPT_INJECTION,
                success_rate=0.1,
                vulnerabilities={"items": []},
                total_probes=10,
                successful_attacks=1,
            )

        mock_session.add.assert_called_once_with(mock_report)
        mock_session.commit.assert_called_once()
