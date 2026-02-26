"""Kafka event publishing for aumos-testing-harness.

Defines domain events published by this service and a typed publisher wrapper.

Events published:
  - TestSuiteCreated   — after a new test suite is persisted
  - TestRunStarted     — when a run transitions to RUNNING
  - TestRunCompleted   — when a run completes successfully
  - TestRunFailed      — when a run fails with an error
  - VulnerabilityDetected — immediately when a critical red-team finding is made
  - RedTeamCompleted   — when a full red-team assessment finishes

All events include tenant_id and correlation_id for distributed tracing.
"""

import uuid
from datetime import UTC, datetime

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class TestLifecycleEventPublisher:
    """Publisher for aumos-testing-harness domain events.

    Wraps EventPublisher with typed methods for each event produced by this service.
    Ensures consistent event schemas and structured logging on every publish.

    Args:
        publisher: The underlying EventPublisher from aumos-common.
    """

    def __init__(self, publisher: EventPublisher) -> None:
        """Initialise with the shared event publisher.

        Args:
            publisher: Configured EventPublisher instance from aumos-common.
        """
        self._publisher = publisher

    async def publish_suite_created(
        self,
        tenant_id: uuid.UUID,
        suite_id: uuid.UUID,
        suite_name: str,
        suite_type: str,
        correlation_id: str,
    ) -> None:
        """Publish a TestSuiteCreated event to Kafka.

        Args:
            tenant_id: The tenant that owns the suite.
            suite_id: The newly created suite ID.
            suite_name: Human-readable suite name.
            suite_type: Evaluation domain (llm/rag/agent/red-team).
            correlation_id: Request correlation ID for distributed tracing.
        """
        event = {
            "event_type": "TestSuiteCreated",
            "tenant_id": str(tenant_id),
            "suite_id": str(suite_id),
            "suite_name": suite_name,
            "suite_type": suite_type,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.TEST_LIFECYCLE, event)
        logger.info(
            "Published TestSuiteCreated event",
            tenant_id=str(tenant_id),
            suite_id=str(suite_id),
        )

    async def publish_run_started(
        self,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
        suite_id: uuid.UUID,
        correlation_id: str,
    ) -> None:
        """Publish a TestRunStarted event to Kafka.

        Args:
            tenant_id: The tenant owning the run.
            run_id: The newly started run ID.
            suite_id: The suite being executed.
            correlation_id: Request correlation ID.
        """
        event = {
            "event_type": "TestRunStarted",
            "tenant_id": str(tenant_id),
            "run_id": str(run_id),
            "suite_id": str(suite_id),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.TEST_LIFECYCLE, event)
        logger.info(
            "Published TestRunStarted event",
            tenant_id=str(tenant_id),
            run_id=str(run_id),
        )

    async def publish_run_completed(
        self,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
        suite_id: uuid.UUID,
        passed: int,
        total: int,
        aggregate_score: float,
        correlation_id: str,
    ) -> None:
        """Publish a TestRunCompleted event to Kafka.

        Args:
            tenant_id: The tenant owning the run.
            run_id: The completed run ID.
            suite_id: The executed suite ID.
            passed: Number of metrics that passed the threshold.
            total: Total number of metrics evaluated.
            aggregate_score: Overall aggregate score (0.0-1.0).
            correlation_id: Request correlation ID.
        """
        event = {
            "event_type": "TestRunCompleted",
            "tenant_id": str(tenant_id),
            "run_id": str(run_id),
            "suite_id": str(suite_id),
            "passed": passed,
            "total": total,
            "aggregate_score": aggregate_score,
            "pass_rate": round(passed / max(total, 1), 4),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.TEST_LIFECYCLE, event)
        logger.info(
            "Published TestRunCompleted event",
            tenant_id=str(tenant_id),
            run_id=str(run_id),
            aggregate_score=aggregate_score,
        )

    async def publish_run_failed(
        self,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
        error_message: str,
        correlation_id: str,
    ) -> None:
        """Publish a TestRunFailed event to Kafka.

        Args:
            tenant_id: The tenant owning the run.
            run_id: The failed run ID.
            error_message: Human-readable error description (no sensitive data).
            correlation_id: Request correlation ID.
        """
        event = {
            "event_type": "TestRunFailed",
            "tenant_id": str(tenant_id),
            "run_id": str(run_id),
            "error_message": error_message,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.TEST_LIFECYCLE, event)
        logger.warning(
            "Published TestRunFailed event",
            tenant_id=str(tenant_id),
            run_id=str(run_id),
        )

    async def publish_vulnerability_detected(
        self,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
        attack_type: str,
        success_rate: float,
        severity: str,
        correlation_id: str,
    ) -> None:
        """Publish a VulnerabilityDetected event to Kafka immediately on critical finding.

        This event is published in real-time (not waiting for the full red-team run)
        so that the governance-engine and ci-cd-pipeline can act immediately.

        Args:
            tenant_id: The tenant owning the run.
            run_id: The red-team run ID.
            attack_type: OWASP LLM attack category that was breached.
            success_rate: Fraction of probes that succeeded.
            severity: Risk severity level: critical, high, medium, low.
            correlation_id: Request correlation ID.
        """
        event = {
            "event_type": "VulnerabilityDetected",
            "tenant_id": str(tenant_id),
            "run_id": str(run_id),
            "attack_type": attack_type,
            "success_rate": success_rate,
            "severity": severity,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.RED_TEAM_LIFECYCLE, event)
        logger.warning(
            "Published VulnerabilityDetected event",
            tenant_id=str(tenant_id),
            run_id=str(run_id),
            attack_type=attack_type,
            success_rate=success_rate,
            severity=severity,
        )

    async def publish_red_team_completed(
        self,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
        categories_tested: int,
        total_vulnerabilities: int,
        critical_found: bool,
        correlation_id: str,
    ) -> None:
        """Publish a RedTeamCompleted event to Kafka.

        Args:
            tenant_id: The tenant owning the run.
            run_id: The red-team run ID.
            categories_tested: Number of OWASP categories probed.
            total_vulnerabilities: Total vulnerabilities found across all categories.
            critical_found: True if any category had a success rate > 0.5.
            correlation_id: Request correlation ID.
        """
        event = {
            "event_type": "RedTeamCompleted",
            "tenant_id": str(tenant_id),
            "run_id": str(run_id),
            "categories_tested": categories_tested,
            "total_vulnerabilities": total_vulnerabilities,
            "critical_found": critical_found,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(Topics.RED_TEAM_LIFECYCLE, event)
        logger.info(
            "Published RedTeamCompleted event",
            tenant_id=str(tenant_id),
            run_id=str(run_id),
            critical_found=critical_found,
        )


__all__ = ["TestLifecycleEventPublisher"]
