"""Service-specific settings for aumos-testing-harness.

All standard AumOS configuration is inherited from AumOSSettings.
Repo-specific settings use the AUMOS_TESTHARNESS_ environment variable prefix.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Settings for the aumos-testing-harness service.

    Inherits all standard AumOS settings (database, kafka, keycloak, etc.)
    and adds testing-harness-specific configuration.

    Environment variable prefix: AUMOS_TESTHARNESS_
    """

    service_name: str = "aumos-testing-harness"

    # --- LLM Provider Configuration ---
    openai_api_key: str = Field(default="", description="OpenAI API key for evaluation LLM calls")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model used as evaluation judge")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (alternative evaluation LLM)")
    anthropic_model: str = Field(default="claude-opus-4-6", description="Anthropic model for evaluation")
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_deployment: str = Field(default="gpt-4o", description="Azure OpenAI deployment name")

    # --- Evaluation Framework Settings ---
    default_pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default pass threshold for LLM metrics (0.0-1.0)",
    )
    max_eval_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent evaluation workers",
    )
    eval_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Per-evaluation-run timeout in seconds",
    )
    ragas_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of test cases evaluated per RAGAS batch",
    )

    # --- Red-Team / OWASP Settings ---
    garak_enabled: bool = Field(default=True, description="Enable Garak red-team probe runner")
    garak_probes: str = Field(
        default="encoding,knownbadsignatures,leakage,packagehallucination",
        description="Comma-separated list of Garak probe modules to run",
    )
    giskard_enabled: bool = Field(default=True, description="Enable Giskard vulnerability scanner")
    red_team_max_attempts: int = Field(
        default=50,
        ge=1,
        description="Maximum attack attempts per red-team category",
    )
    red_team_report_retention_days: int = Field(
        default=90,
        ge=1,
        description="Days to retain red-team reports before archival",
    )

    # --- Agent Evaluation Settings ---
    agent_eval_timeout_seconds: int = Field(
        default=600,
        ge=60,
        description="Timeout for agent evaluation runs in seconds",
    )
    agent_max_steps: int = Field(
        default=20,
        ge=1,
        description="Maximum agent steps before forced termination in evaluation",
    )

    # --- Artifact Storage ---
    artifact_bucket: str = Field(default="aumos-test-artifacts", description="S3 bucket for test artifacts")
    artifact_bucket_region: str = Field(default="us-east-1", description="S3 bucket region")
    aws_access_key_id: str = Field(default="", description="AWS access key for artifact storage")
    aws_secret_access_key: str = Field(default="", description="AWS secret key for artifact storage")

    # --- CI/CD Integration ---
    webhook_secret: str = Field(default="", description="HMAC secret for CI/CD webhook verification")
    github_token: str = Field(default="", description="GitHub token for CI status reporting")

    model_config = SettingsConfigDict(env_prefix="AUMOS_TESTHARNESS_")
