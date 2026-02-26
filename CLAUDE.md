# CLAUDE.md — AumOS Testing Harness

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-testing-harness`) is part of **Tier 3: Quality
& Compliance**: comprehensive AI evaluation, red-teaming, and CI/CD gating for all AumOS
AI workloads.

**Release Tier:** A: Fully Open
**Product Mapping:** Product 9 — AI Quality & Safety (cross-cutting concern)
**Phase:** 4 (Months 18-24)

## Repo Purpose

`aumos-testing-harness` provides a unified evaluation framework for all AI systems in
the AumOS platform. It runs automated LLM quality evaluations (14+ metrics), RAG pipeline
validation using RAGAS, multi-step agent capability testing, and OWASP LLM Top 10
red-team assessments using Garak and Giskard. Every other AumOS repo that exposes an
AI capability integrates with this service for pre-merge and continuous quality gating.

## Architecture Position

```
aumos-common ──────────────────────────────────────────────┐
aumos-proto  ──────────────────────────────────────────────┤
aumos-llm-serving ─────────────────────────────────────────┤
aumos-agent-framework ─────────────────────────────────────┼──► aumos-testing-harness
aumos-data-layer ──────────────────────────────────────────┤         │
aumos-observability ───────────────────────────────────────┘         │
                                                                      ▼
                                                            ├── aumos-ci-cd-pipeline (gates)
                                                            ├── aumos-governance-engine (audit)
                                                            ├── aumos-observability (metrics)
                                                            └── aumos-mlops-lifecycle (model cards)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events

**Downstream dependents (other repos IMPORT from this):**
- `aumos-ci-cd-pipeline` — quality gate evaluation results for merge decisions
- `aumos-governance-engine` — test results and red-team reports for compliance audit trails
- `aumos-mlops-lifecycle` — evaluation scores for model card generation
- `aumos-observability` — metric streams from evaluation runs

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| ragas | 0.1.7+ | RAG evaluation (faithfulness, relevancy, context precision/recall) |
| deepeval | 0.21.0+ | LLM evaluation metrics framework |
| openai | 1.14+ | OpenAI API client for evaluation LLM calls |
| langchain-core | 0.1+ | LangChain abstractions for evaluator chaining |
| numpy / pandas | latest | Metric aggregation and statistical analysis |
| scikit-learn | 1.4+ | Scoring utilities |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app

   # WRONG — never reimplement these
   # from jose import jwt
   # from sqlalchemy import create_engine
   # import logging
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**
   ```python
   from aumos_common.events import EventPublisher, Topics

   # Publish after: suite created, run started, run completed, red-team report ready
   await self.publisher.publish(Topics.TEST_LIFECYCLE, TestRunCompletedEvent(...))
   ```

7. **Async by default.** All I/O operations — including evaluator calls — must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure Convention

```
src/aumos_testing_harness/
├── __init__.py
├── main.py                   # FastAPI app entry point using create_app()
├── settings.py               # Extends AumOSSettings with AUMOS_TESTHARNESS_ prefix
├── api/
│   ├── __init__.py
│   ├── router.py             # All endpoints — thin layer delegating to services
│   └── schemas.py            # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── models.py             # SQLAlchemy ORM (thr_ prefix tables)
│   ├── interfaces.py         # Protocol classes for DI
│   └── services.py           # LLMEvalService, RAGEvalService, AgentEvalService, RedTeamService
└── adapters/
    ├── __init__.py
    ├── repositories.py       # TestSuiteRepository, TestRunRepository, etc.
    ├── kafka.py              # TestLifecycleEventPublisher
    ├── llm_evaluator.py      # 14+ LLM metrics via deepeval
    ├── rag_evaluator.py      # RAGAS: faithfulness, relevancy, context precision/recall
    ├── agent_evaluator.py    # Task completion, tool accuracy, multi-step reasoning
    └── red_team_runner.py    # Garak + Giskard OWASP LLM Top 10
```

### Database Table Prefix

All tables in this repo use the `thr_` prefix (Testing HaRness):
- `thr_test_suites`
- `thr_test_runs`
- `thr_test_results`
- `thr_red_team_reports`

### ORM Models

All tenant-scoped tables extend `AumOSModel` from aumos-common which provides:
- `id`: UUID primary key
- `tenant_id`: UUID (RLS-enforced)
- `created_at`: datetime
- `updated_at`: datetime

### Kafka Events Published

This service publishes to the following topics (use `Topics.*` constants):
- `Topics.TEST_LIFECYCLE` — TestSuiteCreated, TestRunStarted, TestRunCompleted, TestRunFailed
- `Topics.RED_TEAM_LIFECYCLE` — RedTeamStarted, RedTeamCompleted, VulnerabilityDetected

Always include `tenant_id` and `correlation_id` in every event payload.

## API Conventions

- All endpoints under `/api/v1/` prefix
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Request ID: `X-Request-ID` header (auto-generated if missing)
- Pagination: `?page=1&page_size=20&sort_by=created_at&sort_order=desc`
- Errors: Standard `ErrorResponse` from aumos-common
- Content-Type: `application/json` (always)

### Endpoint Summary

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/v1/suites | Create a test suite |
| GET | /api/v1/suites | List test suites (paginated) |
| GET | /api/v1/suites/{id} | Get a test suite |
| POST | /api/v1/suites/{id}/run | Execute a test suite |
| GET | /api/v1/runs | List test runs (paginated) |
| GET | /api/v1/runs/{id}/results | Get results for a run |
| POST | /api/v1/red-team | Launch a red-team assessment |
| GET | /api/v1/red-team/{id}/report | Get a red-team report |
| GET | /api/v1/metrics | List available metrics catalog |

## Evaluator Notes

### LLM Evaluation (deepeval)
14 metrics implemented in `adapters/llm_evaluator.py`:
1. Accuracy (GEval) — factual correctness against ground truth
2. Coherence (GEval) — logical flow and consistency
3. Faithfulness — output grounded in context
4. Answer Relevancy — relevance to the question
5. Contextual Precision — relevant context ranked higher
6. Contextual Recall — all relevant context retrieved
7. Contextual Relevancy — context matched to query
8. Hallucination Detection — unsupported claims
9. Toxicity — harmful/offensive content
10. Bias Detection — demographic or ideological bias
11. Summarization Quality — fidelity and coverage
12. Task Completion — goal achievement
13. Tool Call Accuracy — correct tool selection and arguments
14. Latency Score — response time relative to threshold

### RAG Evaluation (RAGAS)
Implemented in `adapters/rag_evaluator.py`:
- Faithfulness — answer grounded in retrieved context
- Answer Relevancy — answer addresses the question
- Context Precision — proportion of relevant context chunks
- Context Recall — coverage of ground truth in retrieved context
- Answer Correctness — end-to-end factual accuracy

### Agent Evaluation
Implemented in `adapters/agent_evaluator.py`:
- Task Completion Rate — fraction of goal criteria met
- Tool Usage Accuracy — correct tool called with correct args
- Multi-Step Reasoning — logical chaining across steps
- Efficiency Score — steps taken vs. optimal path

### Red-Team (OWASP LLM Top 10)
Implemented in `adapters/red_team_runner.py`:
- LLM01: Prompt Injection (Garak encoding + Giskard injection probes)
- LLM02: Insecure Output Handling
- LLM03: Training Data Poisoning detection
- LLM04: Model Denial of Service (rate/complexity probes)
- LLM05: Supply Chain Vulnerability checks
- LLM06: Sensitive Information Disclosure (Garak leakage probes)
- LLM07: Insecure Plugin Design
- LLM08: Excessive Agency (tool-use boundary testing)
- LLM09: Overreliance (confidence calibration)
- LLM10: Model Theft (extraction probes)

**LICENSE NOTE**: Garak is Apache 2.0. Giskard is Apache 2.0. Both are safe to use.
DeepEval is MIT. RAGAS is MIT. All dependencies are license-compliant.

## Environment Variables

All standard env vars are defined in `aumos_common.config.AumOSSettings`.
Repo-specific vars use the prefix `AUMOS_TESTHARNESS_`.
Copy `.env.example` and customize for local development.

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.** JWT, tenant context, DB sessions,
   Kafka publishing, error handling, logging, health checks, pagination — all imported.
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT add GPL/AGPL dependencies.** All eval frameworks here are Apache/MIT licensed.
8. **Do NOT put business logic in API routes.** Routes call services only.
9. **Do NOT create new exception classes** unless they map to a new ErrorCode in aumos-common.
10. **Do NOT bypass RLS.** Cross-tenant eval data access requires explicit approval.
11. **Do NOT call LLM evaluators synchronously.** All evaluator calls must be async or
    wrapped in `asyncio.to_thread()` if the library is sync-only.
12. **Do NOT store raw LLM responses in logs.** They may contain PII or sensitive data.
