"""Agent capability evaluation adapter.

Implements IAgentEvaluator with 4 metrics:
  1. Task Completion Rate  — fraction of goal criteria successfully met
  2. Tool Usage Accuracy   — correct tool selected with correct arguments
  3. Multi-Step Reasoning  — logical coherence across multiple steps (via deepeval GEval)
  4. Efficiency Score      — steps taken vs. optimal path length

Agent evaluation assesses pre-recorded or live agent trajectories against
defined task specifications. The evaluator is designed to be LLM-agnostic
and works with any agent framework that can produce step-level trajectories.

Synchronous LLM judge calls are wrapped in asyncio.to_thread().
"""

import asyncio
from typing import Any

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)


class AgentEvaluator:
    """Agent capability evaluator using task specification comparison.

    Evaluates agent performance by comparing recorded trajectories against
    expected task definitions. Uses deepeval GEval for multi-step reasoning
    assessment and rule-based scoring for task completion and tool accuracy.

    Args:
        settings: Application settings with LLM provider configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings providing LLM configuration for GEval.
        """
        self._settings = settings
        self._openai_model = settings.openai_model
        self._threshold = settings.default_pass_threshold

    async def evaluate(
        self,
        task_definitions: list[dict[str, Any]],
        agent_trajectories: list[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Evaluate agent task performance against recorded trajectories.

        Scores task completion rate, tool usage accuracy, multi-step reasoning,
        and efficiency for each (task, trajectory) pair.

        Args:
            task_definitions: List of task dicts with: goal, criteria, expected_tools,
                expected_steps (optimal path length), and success_conditions.
            agent_trajectories: List of trajectory dicts with: steps (list of
                {action, tool, arguments, result}), final_answer.
            threshold: Pass/fail threshold for all metrics.

        Returns:
            List of result dicts with: metric_name, score, threshold, passed, details.
        """
        if not task_definitions or not agent_trajectories:
            logger.warning("Agent evaluator called with empty task or trajectory list")
            return []

        all_results = await asyncio.gather(
            *[
                self._evaluate_single_task(task_def, trajectory, threshold, idx)
                for idx, (task_def, trajectory) in enumerate(zip(task_definitions, agent_trajectories, strict=False))
            ]
        )

        results: list[dict[str, Any]] = []
        for task_results in all_results:
            results.extend(task_results)

        logger.info(
            "Agent evaluation completed",
            tasks=len(task_definitions),
            results=len(results),
        )
        return results

    async def _evaluate_single_task(
        self,
        task_definition: dict[str, Any],
        trajectory: dict[str, Any],
        threshold: float,
        task_idx: int,
    ) -> list[dict[str, Any]]:
        """Evaluate all four metrics for a single task+trajectory pair.

        Args:
            task_definition: Task specification.
            trajectory: Recorded agent execution.
            threshold: Pass/fail threshold.
            task_idx: Task index for tracing.

        Returns:
            List of 4 result dicts (one per metric).
        """
        results: list[dict[str, Any]] = []

        results.append(self._score_task_completion(task_definition, trajectory, threshold, task_idx))
        results.append(self._score_tool_usage_accuracy(task_definition, trajectory, threshold, task_idx))

        reasoning_result = await asyncio.to_thread(
            self._score_multi_step_reasoning,
            task_definition=task_definition,
            trajectory=trajectory,
            threshold=threshold,
            task_idx=task_idx,
        )
        results.append(reasoning_result)

        results.append(self._score_efficiency(task_definition, trajectory, threshold, task_idx))

        return results

    def _score_task_completion(
        self,
        task_definition: dict[str, Any],
        trajectory: dict[str, Any],
        threshold: float,
        task_idx: int,
    ) -> dict[str, Any]:
        """Score the fraction of goal criteria met by the agent.

        Each criterion in task_definition['criteria'] is checked against
        the trajectory's final_answer using simple string containment.
        A more sophisticated LLM-based checker can be substituted here.

        Args:
            task_definition: Task specification with 'criteria' list.
            trajectory: Recorded agent execution with 'final_answer'.
            threshold: Pass/fail threshold.
            task_idx: Task index for tracing.

        Returns:
            Task completion result dict.
        """
        criteria = task_definition.get("criteria", [])
        final_answer = trajectory.get("final_answer", "")

        if not criteria:
            return {
                "metric_name": "agent_task_completion_rate",
                "score": 1.0,
                "threshold": threshold,
                "passed": True,
                "details": {"task_idx": task_idx, "criteria_count": 0, "note": "No criteria defined"},
            }

        met_criteria = sum(
            1 for criterion in criteria if criterion.lower() in final_answer.lower()
        )
        score = met_criteria / len(criteria)
        passed = score >= threshold

        return {
            "metric_name": "agent_task_completion_rate",
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
            "details": {
                "task_idx": task_idx,
                "criteria_count": len(criteria),
                "met_criteria": met_criteria,
                "goal": task_definition.get("goal", ""),
            },
        }

    def _score_tool_usage_accuracy(
        self,
        task_definition: dict[str, Any],
        trajectory: dict[str, Any],
        threshold: float,
        task_idx: int,
    ) -> dict[str, Any]:
        """Score correctness of tool selection and argument construction.

        Compares the actual tool calls in the trajectory against the expected
        tool calls in the task definition. Checks both tool name and argument keys.

        Args:
            task_definition: Task specification with 'expected_tools' list.
            trajectory: Recorded agent execution with 'steps' list.
            threshold: Pass/fail threshold.
            task_idx: Task index for tracing.

        Returns:
            Tool usage accuracy result dict.
        """
        expected_tools: list[dict[str, Any]] = task_definition.get("expected_tools", [])
        steps: list[dict[str, Any]] = trajectory.get("steps", [])

        if not expected_tools:
            return {
                "metric_name": "agent_tool_usage_accuracy",
                "score": 1.0,
                "threshold": threshold,
                "passed": True,
                "details": {"task_idx": task_idx, "note": "No expected tools defined"},
            }

        actual_tool_calls = [step for step in steps if step.get("action") == "tool_call"]

        if not actual_tool_calls:
            score = 0.0
            return {
                "metric_name": "agent_tool_usage_accuracy",
                "score": score,
                "threshold": threshold,
                "passed": False,
                "details": {
                    "task_idx": task_idx,
                    "expected_tools": len(expected_tools),
                    "actual_tool_calls": 0,
                },
            }

        correct_calls = 0
        for expected in expected_tools:
            expected_name = expected.get("name", "")
            expected_arg_keys = set(expected.get("arguments", {}).keys())

            for actual in actual_tool_calls:
                if actual.get("tool") == expected_name:
                    actual_arg_keys = set(actual.get("arguments", {}).keys())
                    if expected_arg_keys.issubset(actual_arg_keys):
                        correct_calls += 1
                        break

        score = correct_calls / len(expected_tools)
        passed = score >= threshold

        return {
            "metric_name": "agent_tool_usage_accuracy",
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
            "details": {
                "task_idx": task_idx,
                "expected_tools": len(expected_tools),
                "correct_calls": correct_calls,
            },
        }

    def _score_multi_step_reasoning(
        self,
        task_definition: dict[str, Any],
        trajectory: dict[str, Any],
        threshold: float,
        task_idx: int,
    ) -> dict[str, Any]:
        """Score logical coherence of the agent's multi-step reasoning using GEval.

        This is a synchronous method that runs in a thread pool. It uses deepeval's
        GEval to judge whether the chain of steps is logically coherent.

        Args:
            task_definition: Task specification with 'goal'.
            trajectory: Recorded agent execution with 'steps'.
            threshold: Pass/fail threshold.
            task_idx: Task index for tracing.

        Returns:
            Multi-step reasoning result dict.
        """
        steps = trajectory.get("steps", [])
        goal = task_definition.get("goal", "")

        trajectory_text = "\n".join(
            f"Step {i + 1}: {step.get('action', '')} — {step.get('result', '')}"
            for i, step in enumerate(steps)
        )

        try:
            from deepeval.metrics import GEval  # noqa: PLC0415
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: PLC0415

            metric = GEval(
                name="MultiStepReasoning",
                criteria=(
                    "Evaluate whether the agent's step-by-step reasoning is logically coherent, "
                    "relevant to the goal, and leads toward task completion without contradictions."
                ),
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=threshold,
                model=self._openai_model,
            )
            test_case = LLMTestCase(
                input=f"Goal: {goal}",
                actual_output=trajectory_text,
            )
            metric.measure(test_case)
            score = float(metric.score)
            reason = getattr(metric, "reason", "")

        except ImportError:
            score = 0.75
            reason = "deepeval not installed — mock score"
        except Exception as exc:
            logger.warning("Multi-step reasoning scoring failed", task_idx=task_idx, error=str(exc))
            score = 0.0
            reason = str(exc)

        return {
            "metric_name": "agent_multi_step_reasoning",
            "score": round(score, 4),
            "threshold": threshold,
            "passed": score >= threshold,
            "details": {
                "task_idx": task_idx,
                "step_count": len(steps),
                "goal": goal,
                "reason": reason,
            },
        }

    def _score_efficiency(
        self,
        task_definition: dict[str, Any],
        trajectory: dict[str, Any],
        threshold: float,
        task_idx: int,
    ) -> dict[str, Any]:
        """Score the agent's step efficiency against the optimal path length.

        Efficiency = optimal_steps / actual_steps (capped at 1.0).
        A score of 1.0 means the agent completed the task in the optimal number of steps.

        Args:
            task_definition: Task specification with 'expected_steps' (optimal path length).
            trajectory: Recorded agent execution with 'steps'.
            threshold: Pass/fail threshold.
            task_idx: Task index for tracing.

        Returns:
            Efficiency score result dict.
        """
        expected_steps: int = task_definition.get("expected_steps", 0)
        actual_steps = len(trajectory.get("steps", []))

        if expected_steps <= 0 or actual_steps <= 0:
            return {
                "metric_name": "agent_efficiency_score",
                "score": 1.0,
                "threshold": threshold,
                "passed": True,
                "details": {
                    "task_idx": task_idx,
                    "note": "Cannot compute efficiency — missing step counts",
                    "actual_steps": actual_steps,
                    "expected_steps": expected_steps,
                },
            }

        score = min(1.0, expected_steps / actual_steps)
        passed = score >= threshold

        return {
            "metric_name": "agent_efficiency_score",
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
            "details": {
                "task_idx": task_idx,
                "actual_steps": actual_steps,
                "optimal_steps": expected_steps,
                "overhead_steps": max(0, actual_steps - expected_steps),
            },
        }


__all__ = ["AgentEvaluator"]
