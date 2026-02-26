"""SQLAlchemy repository implementations for aumos-testing-harness.

Repositories extend BaseRepository from aumos-common which provides:
  - Automatic RLS tenant isolation via set_tenant_context
  - Standard CRUD operations (get, list, create, update, delete)
  - Pagination support via paginate()

Implement only the methods that differ from BaseRepository defaults.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext
from aumos_common.database import BaseRepository
from aumos_common.pagination import PageRequest, PageResponse

from aumos_testing_harness.core.interfaces import (
    IRedTeamReportRepository,
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


class TestSuiteRepository(BaseRepository, ITestSuiteRepository):
    """Repository for TestSuite persistence.

    Args:
        session: Async SQLAlchemy session (injected via FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with the database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session, TestSuite)

    async def get_by_id(self, suite_id: uuid.UUID, tenant: TenantContext) -> TestSuite | None:
        """Retrieve a TestSuite by primary key within tenant scope.

        Args:
            suite_id: The suite UUID.
            tenant: Tenant context for RLS isolation.

        Returns:
            The matching TestSuite or None.
        """
        result = await self.session.execute(
            select(TestSuite).where(
                TestSuite.id == str(suite_id),
                TestSuite.tenant_id == str(tenant.tenant_id),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestSuite]:
        """List all suites for a tenant, paginated.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestSuite records.
        """
        query = (
            select(TestSuite)
            .where(TestSuite.tenant_id == str(tenant.tenant_id))
            .order_by(TestSuite.created_at.desc())
        )
        return await self.paginate(query, page)

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
            name: Suite name.
            suite_type: Evaluation domain string.
            config: Suite configuration JSONB payload.
            description: Optional description.

        Returns:
            The newly created TestSuite.
        """
        suite = TestSuite(
            tenant_id=str(tenant.tenant_id),
            name=name,
            suite_type=suite_type,
            config=config,
            description=description,
        )
        self.session.add(suite)
        await self.session.commit()
        await self.session.refresh(suite)
        return suite

    async def delete(self, suite_id: uuid.UUID, tenant: TenantContext) -> None:
        """Soft-delete a TestSuite by marking it inactive.

        Args:
            suite_id: The suite UUID.
            tenant: Tenant context for RLS isolation.
        """
        await self.session.execute(
            update(TestSuite)
            .where(
                TestSuite.id == str(suite_id),
                TestSuite.tenant_id == str(tenant.tenant_id),
            )
            .values(is_active=False)
        )
        await self.session.commit()


class TestRunRepository(BaseRepository, ITestRunRepository):
    """Repository for TestRun persistence.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with the database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session, TestRun)

    async def get_by_id(self, run_id: uuid.UUID, tenant: TenantContext) -> TestRun | None:
        """Retrieve a TestRun by primary key within tenant scope.

        Args:
            run_id: The run UUID.
            tenant: Tenant context for RLS isolation.

        Returns:
            The matching TestRun or None.
        """
        result = await self.session.execute(
            select(TestRun).where(
                TestRun.id == str(run_id),
                TestRun.tenant_id == str(tenant.tenant_id),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, tenant: TenantContext, page: PageRequest) -> PageResponse[TestRun]:
        """List all test runs for a tenant, paginated.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestRun records.
        """
        query = (
            select(TestRun)
            .where(TestRun.tenant_id == str(tenant.tenant_id))
            .order_by(TestRun.created_at.desc())
        )
        return await self.paginate(query, page)

    async def create(
        self,
        tenant: TenantContext,
        suite_id: uuid.UUID,
        ci_build_id: str | None = None,
    ) -> TestRun:
        """Create a TestRun in PENDING status.

        Args:
            tenant: Tenant context for RLS isolation.
            suite_id: The suite being executed.
            ci_build_id: Optional CI build identifier.

        Returns:
            The newly created TestRun.
        """
        run = TestRun(
            tenant_id=str(tenant.tenant_id),
            suite_id=str(suite_id),
            status=RunStatus.PENDING,
            summary={},
            ci_build_id=ci_build_id,
        )
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run

    async def update_status(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
        status: RunStatus,
        summary: dict | None = None,
    ) -> TestRun:
        """Update status and optional summary on an existing TestRun.

        Args:
            run_id: The run UUID.
            tenant: Tenant context for RLS isolation.
            status: New lifecycle status.
            summary: Optional aggregated summary to store.

        Returns:
            The updated TestRun.
        """
        values: dict = {"status": status}
        if summary is not None:
            values["summary"] = summary
        if status == RunStatus.RUNNING:
            values["started_at"] = datetime.now(UTC)
        if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            values["completed_at"] = datetime.now(UTC)

        await self.session.execute(
            update(TestRun)
            .where(
                TestRun.id == str(run_id),
                TestRun.tenant_id == str(tenant.tenant_id),
            )
            .values(**values)
        )
        await self.session.commit()

        updated = await self.get_by_id(run_id, tenant)
        assert updated is not None  # noqa: S101
        return updated


class TestResultRepository(BaseRepository, ITestResultRepository):
    """Repository for TestResult persistence.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with the database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session, TestResult)

    async def list_by_run(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
        page: PageRequest,
    ) -> PageResponse[TestResult]:
        """List all results for a given run, paginated.

        Args:
            run_id: The parent run UUID.
            tenant: Tenant context for RLS isolation.
            page: Pagination parameters.

        Returns:
            Paginated list of TestResult records.
        """
        query = (
            select(TestResult)
            .where(
                TestResult.run_id == str(run_id),
                TestResult.tenant_id == str(tenant.tenant_id),
            )
            .order_by(TestResult.metric_name)
        )
        return await self.paginate(query, page)

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
            results: List of result dicts with metric_name, score, threshold, passed, details.

        Returns:
            List of created TestResult records.
        """
        created: list[TestResult] = []
        for result_data in results:
            result = TestResult(
                tenant_id=str(tenant.tenant_id),
                run_id=str(run_id),
                metric_name=result_data["metric_name"],
                score=result_data["score"],
                threshold=result_data["threshold"],
                passed=result_data["passed"],
                details=result_data.get("details", {}),
            )
            self.session.add(result)
            created.append(result)

        await self.session.commit()
        for result in created:
            await self.session.refresh(result)
        return created


class RedTeamReportRepository(BaseRepository, IRedTeamReportRepository):
    """Repository for RedTeamReport persistence.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with the database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session, RedTeamReport)

    async def get_by_run(
        self,
        run_id: uuid.UUID,
        tenant: TenantContext,
    ) -> list[RedTeamReport]:
        """Retrieve all red-team reports for a run.

        Args:
            run_id: The parent run UUID.
            tenant: Tenant context for RLS isolation.

        Returns:
            List of RedTeamReport records.
        """
        result = await self.session.execute(
            select(RedTeamReport)
            .where(
                RedTeamReport.run_id == str(run_id),
                RedTeamReport.tenant_id == str(tenant.tenant_id),
            )
            .order_by(RedTeamReport.attack_type)
        )
        return list(result.scalars().all())

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
            success_rate: Fraction of probes that succeeded.
            vulnerabilities: Sanitised vulnerability descriptions.
            total_probes: Total probes attempted.
            successful_attacks: Successful probe count.

        Returns:
            The newly created RedTeamReport.
        """
        report = RedTeamReport(
            tenant_id=str(tenant.tenant_id),
            run_id=str(run_id),
            attack_type=attack_type,
            success_rate=success_rate,
            vulnerabilities=vulnerabilities,
            total_probes=total_probes,
            successful_attacks=successful_attacks,
        )
        self.session.add(report)
        await self.session.commit()
        await self.session.refresh(report)
        return report


__all__ = [
    "TestSuiteRepository",
    "TestRunRepository",
    "TestResultRepository",
    "RedTeamReportRepository",
]
