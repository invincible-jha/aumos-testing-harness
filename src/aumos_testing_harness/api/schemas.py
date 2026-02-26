"""Pydantic request and response schemas for the aumos-testing-harness API.

All API inputs and outputs use Pydantic models — never raw dicts.
Schemas are grouped by resource: TestSuite, TestRun, TestResult, RedTeam, Metrics.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aumos_testing_harness.core.models import MetricName, RedTeamAttackType, RunStatus, SuiteType


# ---------------------------------------------------------------------------
# TestSuite schemas
# ---------------------------------------------------------------------------


class TestSuiteCreateRequest(BaseModel):
    """Request body for creating a new test suite."""

    name: str = Field(min_length=1, max_length=255, description="Human-readable suite name")
    description: str | None = Field(default=None, max_length=1000, description="Optional description")
    suite_type: SuiteType = Field(description="Evaluation domain: llm, rag, agent, or red-team")
    config: dict[str, Any] = Field(
        description=(
            "Suite configuration. For LLM: {metrics, test_cases, threshold, model_endpoint}. "
            "For RAG: {test_cases with question/answer/contexts, threshold}. "
            "For agent: {task_definitions, agent_trajectories, threshold}. "
            "For red-team: {target_endpoint, owasp_categories}."
        )
    )


class TestSuiteResponse(BaseModel):
    """Response schema for a single test suite."""

    id: uuid.UUID = Field(description="Unique suite identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    name: str = Field(description="Suite name")
    description: str | None = Field(description="Optional description")
    suite_type: SuiteType = Field(description="Evaluation domain")
    config: dict[str, Any] = Field(description="Suite configuration")
    is_active: bool = Field(description="Whether this suite is active")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = {"from_attributes": True}


class TestSuiteListResponse(BaseModel):
    """Paginated list of test suites."""

    items: list[TestSuiteResponse] = Field(description="Suite records")
    total: int = Field(description="Total count across all pages")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


# ---------------------------------------------------------------------------
# TestRun schemas
# ---------------------------------------------------------------------------


class RunSuiteRequest(BaseModel):
    """Request body for executing a test suite."""

    ci_build_id: str | None = Field(
        default=None,
        max_length=255,
        description="Optional CI build identifier for correlation (e.g. GitHub Actions run ID)",
    )


class TestRunResponse(BaseModel):
    """Response schema for a single test run."""

    id: uuid.UUID = Field(description="Unique run identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    suite_id: uuid.UUID = Field(description="The suite that was executed")
    status: RunStatus = Field(description="Lifecycle status of the run")
    started_at: datetime | None = Field(description="When the run started executing")
    completed_at: datetime | None = Field(description="When the run finished")
    summary: dict[str, Any] = Field(description="Aggregated scoring summary")
    ci_build_id: str | None = Field(description="Optional CI build identifier")
    created_at: datetime = Field(description="Record creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = {"from_attributes": True}


class TestRunListResponse(BaseModel):
    """Paginated list of test runs."""

    items: list[TestRunResponse] = Field(description="Run records")
    total: int = Field(description="Total count across all pages")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


# ---------------------------------------------------------------------------
# TestResult schemas
# ---------------------------------------------------------------------------


class TestResultResponse(BaseModel):
    """Response schema for a single metric result within a run."""

    id: uuid.UUID = Field(description="Unique result identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    run_id: uuid.UUID = Field(description="Parent test run identifier")
    metric_name: str = Field(description="Name of the evaluated metric")
    score: float = Field(ge=0.0, le=1.0, description="Metric score (0.0–1.0)")
    threshold: float = Field(ge=0.0, le=1.0, description="Pass/fail threshold")
    passed: bool = Field(description="Whether the score met the threshold")
    details: dict[str, Any] = Field(description="Evaluation details including input/output")
    created_at: datetime = Field(description="Evaluation timestamp")

    model_config = {"from_attributes": True}


class TestResultListResponse(BaseModel):
    """Paginated list of test results for a run."""

    items: list[TestResultResponse] = Field(description="Result records")
    total: int = Field(description="Total count across all pages")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


# ---------------------------------------------------------------------------
# Red-team schemas
# ---------------------------------------------------------------------------


class RedTeamLaunchRequest(BaseModel):
    """Request body for launching a red-team assessment."""

    suite_id: uuid.UUID = Field(description="The test suite defining red-team configuration")
    target_endpoint: str = Field(
        min_length=1,
        max_length=2048,
        description="URL of the model inference endpoint to probe",
    )
    owasp_categories: list[str] | None = Field(
        default=None,
        description=(
            "OWASP LLM Top 10 categories to test (e.g. ['LLM01', 'LLM06']). "
            "Defaults to all 10 categories if not specified."
        ),
    )
    max_attempts_per_category: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum probe attempts per attack category",
    )


class RedTeamReportResponse(BaseModel):
    """Response schema for a single red-team attack report."""

    id: uuid.UUID = Field(description="Unique report identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    run_id: uuid.UUID = Field(description="Parent test run identifier")
    attack_type: RedTeamAttackType = Field(description="OWASP LLM attack category")
    success_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of attack probes that succeeded (0.0 = fully resilient)",
    )
    vulnerabilities: dict[str, Any] = Field(
        description="Sanitised vulnerability descriptions and recommended mitigations"
    )
    total_probes: int = Field(description="Total number of attack probes attempted")
    successful_attacks: int = Field(description="Number of successful attack probes")
    created_at: datetime = Field(description="Report generation timestamp")

    model_config = {"from_attributes": True}


class RedTeamReportListResponse(BaseModel):
    """All red-team reports for a completed assessment run."""

    run_id: uuid.UUID = Field(description="The test run containing these reports")
    reports: list[RedTeamReportResponse] = Field(description="Individual attack category reports")
    summary: dict[str, Any] = Field(description="Aggregated vulnerability summary")


# ---------------------------------------------------------------------------
# Metrics catalog schemas
# ---------------------------------------------------------------------------


class MetricInfo(BaseModel):
    """Information about a single supported evaluation metric."""

    name: MetricName = Field(description="Machine-readable metric identifier")
    display_name: str = Field(description="Human-readable metric name")
    category: str = Field(description="Metric category: llm, rag, or agent")
    requires_ground_truth: bool = Field(description="Whether a ground truth reference is required")
    description: str = Field(description="What the metric measures")
    score_range: str = Field(description="Score range (always 0.0–1.0)")
    framework: str = Field(description="Underlying evaluation framework")


class MetricsCatalogResponse(BaseModel):
    """Full catalog of available evaluation metrics."""

    metrics: list[MetricInfo] = Field(description="All supported metrics")
    total: int = Field(description="Total number of supported metrics")


__all__ = [
    "TestSuiteCreateRequest",
    "TestSuiteResponse",
    "TestSuiteListResponse",
    "RunSuiteRequest",
    "TestRunResponse",
    "TestRunListResponse",
    "TestResultResponse",
    "TestResultListResponse",
    "RedTeamLaunchRequest",
    "RedTeamReportResponse",
    "RedTeamReportListResponse",
    "MetricInfo",
    "MetricsCatalogResponse",
]
