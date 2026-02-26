# aumos-testing-harness

Comprehensive AI testing framework for the AumOS Enterprise platform. Provides automated
LLM quality evaluation, RAG pipeline validation, multi-step agent capability testing, and
OWASP LLM Top 10 red-team assessments — all integrated into CI/CD pipelines.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [API Reference](#5-api-reference)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Red-Team Assessment](#7-red-team-assessment)
8. [CI/CD Integration](#8-cicd-integration)
9. [Development Guide](#9-development-guide)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

`aumos-testing-harness` (repo #24) is the centralized AI quality gate for all AumOS AI
workloads. It evaluates correctness, safety, and robustness of LLM responses, RAG
pipelines, and autonomous agents across every tenant in the platform.

**Key capabilities:**

- **14+ LLM metrics** — accuracy, coherence, faithfulness, hallucination, toxicity, bias,
  summarization quality, task completion, tool call accuracy, and more
- **RAGAS RAG evaluation** — faithfulness, answer relevancy, context precision/recall,
  answer correctness
- **Agent evaluation** — task completion rate, tool usage accuracy, multi-step reasoning,
  efficiency score
- **OWASP LLM Top 10 red-teaming** — automated attack probes using Garak and Giskard
  covering all 10 OWASP LLM vulnerability categories
- **CI/CD gating** — webhook-based result publishing to `aumos-ci-cd-pipeline` for
  automated merge gates
- **Multi-tenant isolation** — all test suites and results are RLS-scoped to tenants

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     aumos-testing-harness                        │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ LLM Eval │  │  RAG Eval    │  │Agent Eval │  │ Red Team  │  │
│  │(deepeval)│  │   (RAGAS)    │  │           │  │(Garak +   │  │
│  │ 14 metrics│  │ 5 metrics   │  │4 metrics  │  │ Giskard)  │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └─────┬─────┘  │
│       └───────────────┴──────────────── ┴──────────────┘        │
│                              │                                   │
│                    ┌─────────▼──────────┐                        │
│                    │  Core Services     │                        │
│                    │  (DI, orchestration│                        │
│                    └─────────┬──────────┘                        │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │ PostgreSQL  │  │    Kafka        │  │   FastAPI REST   │    │
│  │(thr_ tables)│  │ (test events)   │  │    /api/v1/      │    │
│  └─────────────┘  └─────────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
        ▲                    │                    │
        │                    ▼                    ▼
aumos-common        aumos-ci-cd-pipeline   aumos-governance-engine
aumos-proto         aumos-observability    aumos-mlops-lifecycle
```

**Dependency graph:**
- Upstream: `aumos-common`, `aumos-proto`
- Downstream: `aumos-ci-cd-pipeline`, `aumos-governance-engine`, `aumos-mlops-lifecycle`,
  `aumos-observability`

---

## 3. Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to AumOS internal PyPI (for `aumos-common` and `aumos-proto`)

### Local Development

```bash
# Clone and enter the repository
git clone https://github.com/aumos-enterprise/aumos-testing-harness.git
cd aumos-testing-harness

# Install all dependencies including dev tools
make install

# Configure environment
cp .env.example .env
# Edit .env — set your AUMOS_TESTHARNESS_OPENAI_API_KEY and database URL

# Start local infrastructure (Postgres, Redis, Kafka)
make docker-run

# Run the service
uvicorn aumos_testing_harness.main:app --reload --port 8001
```

### Verify the Setup

```bash
# Health check
curl http://localhost:8001/live
curl http://localhost:8001/ready

# List available metrics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8001/api/v1/metrics
```

---

## 4. Configuration

All configuration uses the `AUMOS_TESTHARNESS_` prefix. Standard AumOS settings
(database, Kafka, Keycloak) are inherited from `aumos-common`.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_TESTHARNESS_OPENAI_API_KEY` | — | OpenAI API key for evaluation LLM |
| `AUMOS_TESTHARNESS_OPENAI_MODEL` | `gpt-4o` | Model used for evaluation judgments |
| `AUMOS_TESTHARNESS_DEFAULT_PASS_THRESHOLD` | `0.7` | Score threshold for pass/fail |
| `AUMOS_TESTHARNESS_MAX_EVAL_WORKERS` | `4` | Concurrent evaluation workers |
| `AUMOS_TESTHARNESS_EVAL_TIMEOUT_SECONDS` | `300` | Per-run evaluation timeout |
| `AUMOS_TESTHARNESS_RAGAS_BATCH_SIZE` | `10` | RAGAS evaluation batch size |
| `AUMOS_TESTHARNESS_GARAK_ENABLED` | `true` | Enable Garak red-team probes |
| `AUMOS_TESTHARNESS_GISKARD_ENABLED` | `true` | Enable Giskard vulnerability scans |
| `AUMOS_TESTHARNESS_RED_TEAM_MAX_ATTEMPTS` | `50` | Max attack attempts per category |

See `.env.example` for the full variable list.

---

## 5. API Reference

All endpoints require Bearer JWT authentication and the `X-Tenant-ID` header.

### Test Suites

```
POST   /api/v1/suites           Create a test suite
GET    /api/v1/suites           List test suites (paginated)
GET    /api/v1/suites/{id}      Get a test suite by ID
POST   /api/v1/suites/{id}/run  Execute a test suite
```

**Create Suite Request:**
```json
{
  "name": "GPT-4o Production QA",
  "suite_type": "llm",
  "config": {
    "model_endpoint": "https://...",
    "metrics": ["accuracy", "faithfulness", "toxicity"],
    "threshold": 0.8,
    "test_cases": [
      {"input": "...", "expected_output": "...", "context": ["..."]}
    ]
  }
}
```

### Test Runs

```
GET    /api/v1/runs             List all test runs
GET    /api/v1/runs/{id}/results  Get results for a completed run
```

### Red-Team

```
POST   /api/v1/red-team         Launch a red-team assessment
GET    /api/v1/red-team/{id}/report  Get a red-team report
```

**Red-Team Request:**
```json
{
  "suite_id": "uuid",
  "target_endpoint": "https://...",
  "owasp_categories": ["LLM01", "LLM06", "LLM08"],
  "max_attempts_per_category": 25
}
```

### Metrics Catalog

```
GET    /api/v1/metrics          List all available evaluation metrics
```

---

## 6. Evaluation Metrics

### LLM Metrics (via deepeval)

| Metric | Requires Ground Truth | Description |
|--------|-----------------------|-------------|
| `accuracy` | Yes | Factual correctness scored by LLM judge |
| `coherence` | No | Logical flow and internal consistency |
| `faithfulness` | No | Output grounded in provided context |
| `answer_relevancy` | No | How well the answer addresses the question |
| `contextual_precision` | Yes | Relevant context ranked higher in retrieval |
| `contextual_recall` | Yes | All relevant information retrieved |
| `contextual_relevancy` | No | Context matched to the query |
| `hallucination` | No | Detects unsupported claims |
| `toxicity` | No | Harmful or offensive content detection |
| `bias` | No | Demographic or ideological bias |
| `summarization` | Yes | Fidelity and coverage of summaries |
| `task_completion` | Yes | Goal achievement against criteria |
| `tool_call_accuracy` | Yes | Correct tool and argument selection |
| `latency_score` | No | Response time vs. configured threshold |

### RAG Metrics (via RAGAS)

| Metric | Description |
|--------|-------------|
| `ragas_faithfulness` | Answer is grounded in retrieved chunks |
| `ragas_answer_relevancy` | Answer addresses the user query |
| `ragas_context_precision` | Relevant chunks ranked above irrelevant ones |
| `ragas_context_recall` | Ground truth information present in context |
| `ragas_answer_correctness` | End-to-end factual accuracy |

### Agent Metrics

| Metric | Description |
|--------|-------------|
| `task_completion_rate` | Fraction of goal criteria successfully met |
| `tool_usage_accuracy` | Correct tool called with correct arguments |
| `multi_step_reasoning` | Logical coherence across multiple steps |
| `efficiency_score` | Steps taken vs. optimal path length |

---

## 7. Red-Team Assessment

The red-team runner implements OWASP LLM Top 10 attack probes using Garak and Giskard.

| OWASP ID | Category | Tooling |
|----------|----------|---------|
| LLM01 | Prompt Injection | Garak + Giskard |
| LLM02 | Insecure Output Handling | Giskard |
| LLM03 | Training Data Poisoning | Giskard |
| LLM04 | Model Denial of Service | Garak |
| LLM05 | Supply Chain Vulnerabilities | Static analysis |
| LLM06 | Sensitive Information Disclosure | Garak leakage probes |
| LLM07 | Insecure Plugin Design | Agent boundary testing |
| LLM08 | Excessive Agency | Tool-use boundary probes |
| LLM09 | Overreliance | Confidence calibration checks |
| LLM10 | Model Theft | Extraction resistance probes |

Red-team reports include: attack type, success rate, sample vulnerabilities (sanitized),
and recommended mitigations.

---

## 8. CI/CD Integration

The testing harness integrates with `aumos-ci-cd-pipeline` via Kafka events and webhooks.

### Webhook Setup

```yaml
# In your CI pipeline YAML
- name: Run AI Quality Gate
  run: |
    curl -X POST $AUMOS_TESTHARNESS_URL/api/v1/suites/$SUITE_ID/run \
      -H "Authorization: Bearer $TOKEN" \
      -H "X-Tenant-ID: $TENANT_ID" \
      -H "Content-Type: application/json" \
      -d '{"ci_build_id": "'$GITHUB_RUN_ID'"}'
```

### Kafka Events

The harness publishes to these topics after run completion:
- `aumos.test.run.completed` — includes pass/fail status and aggregate scores
- `aumos.test.vulnerability.detected` — immediately on red-team finding

### GitHub Actions

A reusable workflow is available in `.github/workflows/ci.yml`. External repos can
reference it to add AI quality gates to their own CI pipelines.

---

## 9. Development Guide

### Running Tests

```bash
make test           # Full test suite with coverage
make test-quick     # Fast run without coverage
make lint           # Ruff linting and format check
make typecheck      # mypy strict type checking
make format         # Auto-format with ruff
```

### Adding a New Metric

1. Add the metric enum value to `core/models.py` (MetricName enum)
2. Implement the scorer in `adapters/llm_evaluator.py` or the relevant evaluator
3. Register the metric in the catalog endpoint in `api/router.py`
4. Add unit tests in `tests/test_services.py`
5. Update the metrics table in this README

### Adding a New Red-Team Category

1. Add the attack type to `RedTeamAttackType` enum in `core/models.py`
2. Implement the probe runner in `adapters/red_team_runner.py`
3. Update the OWASP mapping in `RedTeamService`
4. Add integration tests using mocked probe responses

### Coverage Requirements

- `core/` modules: minimum **80%** coverage
- `adapters/` modules: minimum **60%** coverage
- Run `make test` to verify before opening a PR

---

## 10. Troubleshooting

### Evaluation Times Out

Increase `AUMOS_TESTHARNESS_EVAL_TIMEOUT_SECONDS` and check LLM provider rate limits.
Use `AUMOS_TESTHARNESS_MAX_EVAL_WORKERS=2` to reduce concurrent load.

### RAGAS Scores Are 0.0

Ensure the `context` field in test cases contains the retrieved document chunks, not just
the final answer. RAGAS requires both the question, answer, and source context.

### Garak Probes Fail to Run

Check that `AUMOS_TESTHARNESS_GARAK_ENABLED=true` and that the target endpoint is
reachable from the test harness service network. Use `docker-compose.dev.yml` for local
network connectivity.

### Database Migration Errors

Run migrations in order:
```bash
alembic -c src/aumos_testing_harness/migrations/alembic.ini upgrade head
```

### Red-Team Reports Show No Vulnerabilities

Verify the target model endpoint returns valid JSON. Check `AUMOS_TESTHARNESS_RED_TEAM_MAX_ATTEMPTS`
is not set too low (minimum 10 recommended per category).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 AumOS Enterprise
