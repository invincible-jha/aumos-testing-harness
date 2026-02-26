"""AumOS Testing Harness service entry point.

Initialises the FastAPI application with health checks, database connectivity,
Kafka event publisher, and the evaluation API router.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_testing_harness.api.router import router
from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)
settings = Settings()


async def _check_database() -> bool:
    """Verify database connectivity for the readiness probe.

    Returns:
        True if the database is reachable, False otherwise.
    """
    try:
        from aumos_common.database import get_db_session  # noqa: PLC0415

        async for session in get_db_session():
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
            return True
    except Exception:
        logger.warning("Database health check failed")
        return False
    return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
    - Initialises the async database engine via aumos-common
    - Logs service configuration (non-sensitive fields only)

    On shutdown:
    - Placeholder for Kafka and Redis connection teardown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    logger.info(
        "Starting aumos-testing-harness",
        version="0.1.0",
        openai_model=settings.openai_model,
        default_pass_threshold=settings.default_pass_threshold,
        garak_enabled=settings.garak_enabled,
        giskard_enabled=settings.giskard_enabled,
    )

    init_database(settings.database)
    # TODO: Initialise Kafka EventPublisher for test lifecycle events
    # TODO: Initialise Redis client for run-status caching

    yield

    logger.info("Shutting down aumos-testing-harness")
    # TODO: Close Kafka producer connections
    # TODO: Close Redis connections


app: FastAPI = create_app(
    service_name="aumos-testing-harness",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=_check_database),
    ],
)

app.include_router(router, prefix="/api/v1")
