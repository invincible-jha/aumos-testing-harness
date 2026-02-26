"""Unit tests for aumos-testing-harness core services.

Tests LLMEvalService, RAGEvalService, AgentEvalService, and RedTeamService
with mocked repositories and evaluators.

Coverage target: 80% for core/ modules.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_testing_harness.core.models import RedTeamAttackType, RunStatus, SuiteType
from aumos_testing_harness.core.services import (
    AgentEvalService,
    LLMEvalService,
    RAGEvalService,
    RedTeamService,
)


class TestLLMEvalService:
    """Tests for LLMEvalService."""

    def _make_service(
        self,
        suite_repo: AsyncMock,
        run_repo: AsyncMock,
        result_repo: AsyncMock,
        evaluator: AsyncMock,
    ) -> LLMEvalService:
        """Build a LLMEvalService with the given mocks.

        Args:
            suite_repo: Mock suite repository.
            run_repo: Mock run repository.
            result_repo: Mock result repository.
            evaluator: Mock LLM evaluator.

        Returns:
            LLMEvalService instance.
        """
        return LLMEvalService(
            suite_repo=suite_repo,
            run_repo=run_repo,
            result_repo=result_repo,
            evaluator=evaluator,
            publisher=None,
        )

    @pytest.mark.asyncio()
    async def test_create_suite_calls_repo(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_llm_evaluator: AsyncMock,
    ) -> None:
        """Creating a suite delegates to the repository and returns the suite.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repository.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            mock_llm_evaluator: Mock evaluator.
        """
        service = self._make_service(
            mock_suite_repo, mock_run_repo, mock_result_repo, mock_llm_evaluator
        )

        result = await service.create_suite(
            tenant=mock_tenant,
            name="Test Suite",
            config={"metrics": ["accuracy"], "test_cases": [], "threshold": 0.7},
        )

        mock_suite_repo.create.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio()
    async def test_run_suite_success_path(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_llm_evaluator: AsyncMock,
        sample_suite_id: uuid.UUID,
    ) -> None:
        """A successful suite run updates status to COMPLETED.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repository.
            mock_run_repo: Mock run repository (update_status returns COMPLETED run).
            mock_result_repo: Mock result repository.
            mock_llm_evaluator: Mock evaluator returning passing scores.
            sample_suite_id: Test suite UUID.
        """
        completed_run = MagicMock()
        completed_run.status = RunStatus.COMPLETED
        mock_run_repo.update_status.return_value = completed_run

        service = self._make_service(
            mock_suite_repo, mock_run_repo, mock_result_repo, mock_llm_evaluator
        )

        result = await service.run_suite(mock_tenant, sample_suite_id)

        assert result.status == RunStatus.COMPLETED
        mock_llm_evaluator.evaluate.assert_called_once()
        mock_result_repo.bulk_create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_suite_evaluator_failure_marks_failed(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        sample_suite_id: uuid.UUID,
    ) -> None:
        """When the evaluator raises, the run is marked FAILED.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repository.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            sample_suite_id: Test suite UUID.
        """
        failing_evaluator = AsyncMock()
        failing_evaluator.evaluate.side_effect = RuntimeError("LLM API timeout")

        failed_run = MagicMock()
        failed_run.status = RunStatus.FAILED
        mock_run_repo.update_status.return_value = failed_run

        service = self._make_service(
            mock_suite_repo, mock_run_repo, mock_result_repo, failing_evaluator
        )

        result = await service.run_suite(mock_tenant, sample_suite_id)

        assert result.status == RunStatus.FAILED

    @pytest.mark.asyncio()
    async def test_get_suite_raises_not_found(
        self,
        mock_tenant: MagicMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_llm_evaluator: AsyncMock,
    ) -> None:
        """Getting a non-existent suite raises NotFoundError.

        Args:
            mock_tenant: Mock tenant context.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            mock_llm_evaluator: Mock evaluator.
        """
        from aumos_common.errors import NotFoundError  # noqa: PLC0415

        missing_suite_repo = AsyncMock()
        missing_suite_repo.get_by_id.return_value = None

        service = self._make_service(
            missing_suite_repo, mock_run_repo, mock_result_repo, mock_llm_evaluator
        )

        with pytest.raises(NotFoundError):
            await service.get_suite(mock_tenant, uuid.uuid4())

    @pytest.mark.asyncio()
    async def test_list_suites_returns_paginated(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_llm_evaluator: AsyncMock,
    ) -> None:
        """list_suites delegates to the repo and returns the paginated result.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repository.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            mock_llm_evaluator: Mock evaluator.
        """
        from aumos_common.pagination import PageRequest  # noqa: PLC0415

        service = self._make_service(
            mock_suite_repo, mock_run_repo, mock_result_repo, mock_llm_evaluator
        )

        result = await service.list_suites(mock_tenant, PageRequest(page=1, page_size=20))

        mock_suite_repo.list_all.assert_called_once()
        assert result.total == 1


class TestRAGEvalService:
    """Tests for RAGEvalService."""

    @pytest.mark.asyncio()
    async def test_run_suite_calls_ragas_evaluator(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_rag_evaluator: AsyncMock,
        sample_suite_id: uuid.UUID,
    ) -> None:
        """RAG suite run calls the RAGAS evaluator with extracted test data.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repository with RAG suite config.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            mock_rag_evaluator: Mock RAGAS evaluator.
            sample_suite_id: Test suite UUID.
        """
        rag_suite = MagicMock()
        rag_suite.config = {
            "test_cases": [
                {
                    "question": "What is RAG?",
                    "answer": "Retrieval-Augmented Generation",
                    "contexts": ["RAG combines retrieval with generation."],
                    "ground_truth": "Retrieval-Augmented Generation is a technique...",
                }
            ],
            "threshold": 0.7,
        }
        mock_suite_repo.get_by_id.return_value = rag_suite

        completed_run = MagicMock()
        completed_run.status = RunStatus.COMPLETED
        mock_run_repo.update_status.return_value = completed_run

        service = RAGEvalService(
            suite_repo=mock_suite_repo,
            run_repo=mock_run_repo,
            result_repo=mock_result_repo,
            evaluator=mock_rag_evaluator,
            publisher=None,
        )

        result = await service.run_suite(mock_tenant, sample_suite_id)

        mock_rag_evaluator.evaluate.assert_called_once()
        assert result.status == RunStatus.COMPLETED


class TestAgentEvalService:
    """Tests for AgentEvalService."""

    @pytest.mark.asyncio()
    async def test_run_suite_calls_agent_evaluator(
        self,
        mock_tenant: MagicMock,
        mock_suite_repo: AsyncMock,
        mock_run_repo: AsyncMock,
        mock_result_repo: AsyncMock,
        mock_agent_evaluator: AsyncMock,
        sample_suite_id: uuid.UUID,
    ) -> None:
        """Agent suite run calls the agent evaluator with task definitions.

        Args:
            mock_tenant: Mock tenant context.
            mock_suite_repo: Mock suite repo with agent config.
            mock_run_repo: Mock run repository.
            mock_result_repo: Mock result repository.
            mock_agent_evaluator: Mock agent evaluator.
            sample_suite_id: Test suite UUID.
        """
        agent_suite = MagicMock()
        agent_suite.config = {
            "task_definitions": [
                {
                    "goal": "Retrieve weather for London",
                    "criteria": ["London", "temperature"],
                    "expected_tools": [{"name": "get_weather", "arguments": {"city": "London"}}],
                    "expected_steps": 2,
                }
            ],
            "agent_trajectories": [
                {
                    "steps": [
                        {"action": "tool_call", "tool": "get_weather", "arguments": {"city": "London"}, "result": "15C"},
                    ],
                    "final_answer": "The temperature in London is 15C.",
                }
            ],
            "threshold": 0.7,
        }
        mock_suite_repo.get_by_id.return_value = agent_suite

        completed_run = MagicMock()
        completed_run.status = RunStatus.COMPLETED
        mock_run_repo.update_status.return_value = completed_run

        service = AgentEvalService(
            suite_repo=mock_suite_repo,
            run_repo=mock_run_repo,
            result_repo=mock_result_repo,
            evaluator=mock_agent_evaluator,
            publisher=None,
        )

        result = await service.run_suite(mock_tenant, sample_suite_id)

        mock_agent_evaluator.evaluate.assert_called_once()
        assert result.status == RunStatus.COMPLETED


class TestRedTeamService:
    """Tests for RedTeamService."""

    @pytest.mark.asyncio()
    async def test_launch_assessment_runs_probes(
        self,
        mock_tenant: MagicMock,
        mock_run_repo: AsyncMock,
        mock_report_repo: AsyncMock,
        mock_red_team_runner: AsyncMock,
        sample_suite_id: uuid.UUID,
    ) -> None:
        """Launching an assessment calls the runner for each requested category.

        Args:
            mock_tenant: Mock tenant context.
            mock_run_repo: Mock run repository.
            mock_report_repo: Mock report repository.
            mock_red_team_runner: Mock red-team runner.
            sample_suite_id: Test suite UUID.
        """
        completed_run = MagicMock()
        completed_run.status = RunStatus.COMPLETED
        mock_run_repo.update_status.return_value = completed_run

        service = RedTeamService(
            run_repo=mock_run_repo,
            report_repo=mock_report_repo,
            runner=mock_red_team_runner,
            publisher=None,
        )

        result = await service.launch_assessment(
            tenant=mock_tenant,
            suite_id=sample_suite_id,
            target_endpoint="https://api.example.com/v1/chat",
            owasp_categories=["LLM01", "LLM06"],
            max_attempts_per_category=5,
        )

        mock_red_team_runner.run_probes.assert_called_once()
        assert result.status == RunStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_get_report_raises_not_found_for_missing_run(
        self,
        mock_tenant: MagicMock,
        mock_report_repo: AsyncMock,
        mock_red_team_runner: AsyncMock,
    ) -> None:
        """Getting a report for a non-existent run raises NotFoundError.

        Args:
            mock_tenant: Mock tenant context.
            mock_report_repo: Mock report repository.
            mock_red_team_runner: Mock runner.
        """
        from aumos_common.errors import NotFoundError  # noqa: PLC0415

        missing_run_repo = AsyncMock()
        missing_run_repo.get_by_id.return_value = None

        service = RedTeamService(
            run_repo=missing_run_repo,
            report_repo=mock_report_repo,
            runner=mock_red_team_runner,
            publisher=None,
        )

        with pytest.raises(NotFoundError):
            await service.get_report(mock_tenant, uuid.uuid4())

    def test_resolve_attack_types_defaults_to_all_ten(self) -> None:
        """Passing None for categories returns all 10 OWASP categories."""
        service = RedTeamService(
            run_repo=AsyncMock(),
            report_repo=AsyncMock(),
            runner=AsyncMock(),
            publisher=None,
        )

        result = service._resolve_attack_types(None)

        assert len(result) == 10

    def test_resolve_attack_types_filters_by_category(self) -> None:
        """Passing specific OWASP IDs returns only those categories."""
        service = RedTeamService(
            run_repo=AsyncMock(),
            report_repo=AsyncMock(),
            runner=AsyncMock(),
            publisher=None,
        )

        result = service._resolve_attack_types(["LLM01", "LLM06"])

        assert len(result) == 2
        assert RedTeamAttackType.PROMPT_INJECTION in result
        assert RedTeamAttackType.SENSITIVE_DISCLOSURE in result

    def test_resolve_attack_types_ignores_unknown_categories(self) -> None:
        """Unknown category strings are silently skipped."""
        service = RedTeamService(
            run_repo=AsyncMock(),
            report_repo=AsyncMock(),
            runner=AsyncMock(),
            publisher=None,
        )

        result = service._resolve_attack_types(["LLM01", "UNKNOWN99"])

        assert len(result) == 1
        assert RedTeamAttackType.PROMPT_INJECTION in result
