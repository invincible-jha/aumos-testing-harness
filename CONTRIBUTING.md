# Contributing to aumos-testing-harness

Thank you for contributing to AumOS Enterprise. This guide covers everything you need
to get started and ensure your contributions meet our standards.

## Getting Started

1. Fork the repository (external contributors) or clone directly (AumOS team members)
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```
3. Make your changes following the standards below
4. Submit a pull request targeting `main`

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Docker and Docker Compose
- Access to AumOS internal PyPI (for `aumos-common` and `aumos-proto`)
- An OpenAI API key (or compatible provider) for running evaluator tests

### Install

```bash
# Install all dependencies including dev tools
make install

# Copy and configure environment
cp .env.example .env
# Edit .env — set AUMOS_TESTHARNESS_OPENAI_API_KEY and database URL

# Start local infrastructure
make docker-run
```

### Verify Setup

```bash
make lint       # Should pass with no errors
make typecheck  # Should pass with no errors
make test       # Should pass with coverage >= 80%
```

## Code Standards

All code in this repository must follow the standards defined in [CLAUDE.md](CLAUDE.md).
Key requirements:

- **Type hints on every function** — no exceptions
- **Pydantic models for all API inputs/outputs** — never return raw dicts
- **Structured logging** — use `get_logger(__name__)`, never `print()`
- **Async by default** — all I/O and LLM calls must be async (use `asyncio.to_thread()`
  if the evaluation library is synchronous)
- **Import from aumos-common** — never reimplement shared utilities
- **Google-style docstrings** on all public classes and methods
- **Max line length: 120 characters**

Run `make lint` and `make typecheck` before every commit.

## PR Process

1. Ensure all CI checks pass (lint, typecheck, test, docker build, license check)
2. Fill out the PR template completely
3. Request review from at least one member of `@aumos/platform-team`
4. Squash merge only — keep history clean
5. Delete your branch after merge

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add bias detection metric to LLM evaluator
fix: resolve RAGAS timeout on large context batches
refactor: extract metric threshold logic into MetricConfig model
docs: document OWASP red-team category mapping
test: add integration tests for agent task completion scorer
chore: bump deepeval to 0.22.0
```

Commit messages explain **WHY**, not just what changed.

## Adding New Evaluation Metrics

1. Add the metric name to `MetricName` enum in `src/aumos_testing_harness/core/models.py`
2. Implement the scorer in the appropriate adapter:
   - LLM metrics: `adapters/llm_evaluator.py`
   - RAG metrics: `adapters/rag_evaluator.py`
   - Agent metrics: `adapters/agent_evaluator.py`
3. Register the metric in the catalog list in `api/router.py`
4. Add unit tests with mocked LLM responses
5. Update the metrics table in `README.md`

## Adding New Red-Team Probes

1. Add the attack type to `RedTeamAttackType` enum in `core/models.py`
2. Add the OWASP category mapping to `OWASPCategory` enum
3. Implement the probe runner in `adapters/red_team_runner.py`
4. Update `RedTeamService.run_assessment()` in `core/services.py`
5. Add integration tests with mocked probe responses
6. Update the OWASP table in `README.md`

## License Compliance — CRITICAL

**This is the most important section. Read it carefully.**

AumOS Enterprise is licensed under Apache 2.0. Our enterprise customers have strict
requirements that prohibit AGPL and GPL licensed code in our platform.

### Approved Licenses

The following licenses are approved for dependencies:

- MIT (deepeval, RAGAS)
- Apache Software License 2.0 (Garak, Giskard, OpenTelemetry, FastAPI)
- BSD (2-clause or 3-clause)
- ISC
- Python Software Foundation (PSF)
- Mozilla Public License 2.0 (MPL 2.0) — check with team first

### Checking License Before Adding a Dependency

```bash
pip install pip-licenses
pip install <new-package>
pip-licenses --packages <new-package>
```

If you are unsure about a license, **ask before adding the dependency**.

## Testing Requirements

- All new features must include tests
- Coverage must remain >= 80% for `core/` modules
- Coverage must remain >= 60% for `adapters/`
- Use `testcontainers` for integration tests requiring real infrastructure
- Mock LLM API calls in unit tests (never make live API calls in CI)

```bash
# Run the full test suite
make test

# Run a specific test file
pytest tests/test_services.py -v

# Run with coverage report
pytest tests/ --cov --cov-report=html
```

## Code of Conduct

We are committed to providing a welcoming and respectful environment for all contributors.
All participants are expected to:

- Be respectful and constructive in all interactions
- Focus on what is best for the project and platform
- Accept feedback graciously and provide it thoughtfully
- Report unacceptable behavior to the platform team

Violations may result in removal from the project.
