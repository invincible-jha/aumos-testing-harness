# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding from aumos-repo-template
- FastAPI service with health check endpoints (`/live`, `/ready`)
- SQLAlchemy ORM models with `thr_` prefix: `TestSuite`, `TestRun`, `TestResult`,
  `RedTeamReport`
- `LLMEvalService` with 14 evaluation metrics via deepeval
- `RAGEvalService` with 5 RAGAS metrics (faithfulness, relevancy, context precision/recall,
  answer correctness)
- `AgentEvalService` with task completion, tool accuracy, multi-step reasoning, and
  efficiency scoring
- `RedTeamService` with Garak + Giskard OWASP LLM Top 10 probe runner
- REST API endpoints: suites CRUD, run execution, red-team launch, metrics catalog
- Kafka event publishing for test lifecycle events
- Multi-tenant RLS isolation via aumos-common
- CI/CD pipeline with lint, typecheck, test, Docker build, and license check jobs
- Docker multi-stage build with non-root `aumos` user
- docker-compose.dev.yml with Postgres, Redis, Kafka, Zookeeper
