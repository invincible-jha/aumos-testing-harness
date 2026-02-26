"""SQLAlchemy ORM models for aumos-testing-harness.

All tenant-scoped tables use the thr_ prefix and extend AumOSModel which provides:
  - id: UUID primary key
  - tenant_id: UUID (RLS-enforced)
  - created_at: datetime
  - updated_at: datetime

Tables:
  - thr_test_suites    — reusable test suite definitions
  - thr_test_runs      — individual execution instances of a suite
  - thr_test_results   — per-metric scores within a run
  - thr_red_team_reports — red-team vulnerability assessment results
"""

import enum

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class SuiteType(str, enum.Enum):
    """Classification of test suite by evaluation domain."""

    LLM = "llm"
    RAG = "rag"
    AGENT = "agent"
    RED_TEAM = "red-team"


class RunStatus(str, enum.Enum):
    """Lifecycle status of a test run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricName(str, enum.Enum):
    """Enumeration of all supported evaluation metrics."""

    # LLM metrics (deepeval)
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"
    SUMMARIZATION = "summarization"
    TASK_COMPLETION = "task_completion"
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    LATENCY_SCORE = "latency_score"

    # RAG metrics (RAGAS)
    RAGAS_FAITHFULNESS = "ragas_faithfulness"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    RAGAS_CONTEXT_PRECISION = "ragas_context_precision"
    RAGAS_CONTEXT_RECALL = "ragas_context_recall"
    RAGAS_ANSWER_CORRECTNESS = "ragas_answer_correctness"

    # Agent metrics
    AGENT_TASK_COMPLETION_RATE = "agent_task_completion_rate"
    AGENT_TOOL_USAGE_ACCURACY = "agent_tool_usage_accuracy"
    AGENT_MULTI_STEP_REASONING = "agent_multi_step_reasoning"
    AGENT_EFFICIENCY_SCORE = "agent_efficiency_score"


class RedTeamAttackType(str, enum.Enum):
    """OWASP LLM Top 10 attack categories for red-team assessments."""

    PROMPT_INJECTION = "LLM01_prompt_injection"
    INSECURE_OUTPUT = "LLM02_insecure_output_handling"
    TRAINING_DATA_POISONING = "LLM03_training_data_poisoning"
    MODEL_DOS = "LLM04_model_denial_of_service"
    SUPPLY_CHAIN = "LLM05_supply_chain_vulnerability"
    SENSITIVE_DISCLOSURE = "LLM06_sensitive_information_disclosure"
    INSECURE_PLUGIN = "LLM07_insecure_plugin_design"
    EXCESSIVE_AGENCY = "LLM08_excessive_agency"
    OVERRELIANCE = "LLM09_overreliance"
    MODEL_THEFT = "LLM10_model_theft"


class TestSuite(AumOSModel):
    """A reusable test suite definition.

    Stores the configuration for a named collection of test cases
    targeting a specific evaluation domain (LLM, RAG, agent, or red-team).

    Table: thr_test_suites
    """

    __tablename__ = "thr_test_suites"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    suite_type: Mapped[SuiteType] = mapped_column(
        Enum(SuiteType, name="thr_suite_type_enum"),
        nullable=False,
        index=True,
    )
    # JSONB config holds: model_endpoint, metrics list, threshold, test_cases array
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class TestRun(AumOSModel):
    """A single execution instance of a TestSuite.

    Created when a suite is triggered (manually or via CI). Tracks
    overall run status and aggregated scoring summary.

    Table: thr_test_runs
    """

    __tablename__ = "thr_test_runs"

    suite_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("thr_test_suites.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="thr_run_status_enum"),
        nullable=False,
        default=RunStatus.PENDING,
        index=True,
    )
    started_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # JSONB summary: total_tests, passed, failed, aggregate_score, metrics_summary
    summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # ci_build_id allows CI pipelines to correlate runs to builds
    ci_build_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)


class TestResult(AumOSModel):
    """A single metric score within a test run.

    One TestResult row is created per (run, test_case, metric) triple.
    The passed field is determined by comparing score against threshold.

    Table: thr_test_results
    """

    __tablename__ = "thr_test_results"

    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("thr_test_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # JSONB details: test_case_id, input, expected_output, actual_output, reasoning
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class RedTeamReport(AumOSModel):
    """Results from a red-team attack probe within a test run.

    One row per (run, attack_type). Contains aggregate success rate
    and sanitised vulnerability samples for audit purposes.

    Table: thr_red_team_reports
    """

    __tablename__ = "thr_red_team_reports"

    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("thr_test_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attack_type: Mapped[RedTeamAttackType] = mapped_column(
        Enum(RedTeamAttackType, name="thr_attack_type_enum"),
        nullable=False,
        index=True,
    )
    # success_rate: fraction of attack probes that succeeded (0.0 = fully resilient)
    success_rate: Mapped[float] = mapped_column(Float, nullable=False)
    # JSONB vulnerabilities: list of {probe_id, description, severity, mitigation}
    # Raw attack prompts are NOT stored here — only sanitised descriptions
    vulnerabilities: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    total_probes: Mapped[int] = mapped_column(nullable=False, default=0)
    successful_attacks: Mapped[int] = mapped_column(nullable=False, default=0)


__all__ = [
    "SuiteType",
    "RunStatus",
    "MetricName",
    "RedTeamAttackType",
    "TestSuite",
    "TestRun",
    "TestResult",
    "RedTeamReport",
]
