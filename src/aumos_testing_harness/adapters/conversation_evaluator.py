"""Multi-turn conversation evaluator for aumos-testing-harness.

Evaluates LLM performance across multi-turn conversations using three
orthogonal metrics: coherence, goal completion, and contradiction detection.

GAP-182: Multi-Turn Conversation Evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation.

    Attributes:
        role: Speaker role — 'user' or 'assistant'.
        content: Turn text content.
        turn_index: Zero-based turn position in the conversation.
    """

    role: str
    content: str
    turn_index: int = 0


@dataclass
class MultiTurnTestCase:
    """A complete multi-turn conversation test case.

    Attributes:
        turns: Ordered list of conversation turns.
        goal: High-level objective that the conversation should accomplish.
        expected_final_answer: Optional expected answer at conversation end.
    """

    turns: list[ConversationTurn]
    goal: str
    expected_final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationEvaluator:
    """Evaluates multi-turn conversation quality across three dimensions.

    Metrics:
      - Coherence: Semantic consistency across turns (no topic drift)
      - Goal completion: Whether the conversation achieves its stated objective
      - Contradiction: Detects when assistant contradicts earlier statements

    Args:
        llm_judge: LLM client implementing `async complete(prompt: str) -> str`.
        judge_model: Model identifier for the judge LLM.
    """

    def __init__(self, llm_judge: Any, judge_model: str = "gpt-4o") -> None:
        """Initialise the conversation evaluator.

        Args:
            llm_judge: LLM client for scoring conversations.
            judge_model: Model to use as the judge.
        """
        self._llm_judge = llm_judge
        self._judge_model = judge_model

    async def evaluate(
        self,
        test_case: MultiTurnTestCase,
    ) -> dict[str, Any]:
        """Evaluate a multi-turn conversation across all three metrics.

        Args:
            test_case: The conversation to evaluate.

        Returns:
            Dictionary with coherence, goal_completion, contradiction, and overall scores.
        """
        conversation_text = self._format_conversation(test_case.turns)

        coherence = await self._score_coherence(conversation_text)
        goal_completion = await self._score_goal_completion(conversation_text, test_case.goal)
        contradiction = await self._score_contradiction(conversation_text)

        # Overall score: weighted average (goal completion weighted highest)
        overall = (coherence * 0.3) + (goal_completion * 0.5) + ((1.0 - contradiction) * 0.2)

        result = {
            "coherence": coherence,
            "goal_completion": goal_completion,
            "contradiction_rate": contradiction,
            "overall_score": overall,
            "passed": overall >= 0.7,
            "n_turns": len(test_case.turns),
            "goal": test_case.goal,
            "judge_model": self._judge_model,
        }
        logger.info(
            "conversation_evaluated",
            overall_score=overall,
            n_turns=len(test_case.turns),
        )
        return result

    def _format_conversation(self, turns: list[ConversationTurn]) -> str:
        """Format turns into a readable transcript for the judge.

        Args:
            turns: List of conversation turns.

        Returns:
            Formatted conversation string.
        """
        lines = []
        for turn in turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        return "\n".join(lines)

    async def _score_coherence(self, conversation_text: str) -> float:
        """Score semantic coherence across conversation turns.

        Args:
            conversation_text: Formatted conversation transcript.

        Returns:
            Coherence score in [0, 1].
        """
        prompt = f"""Rate the coherence of this conversation on a scale of 0.0 to 1.0.
Coherence measures whether each response logically follows from the previous turns.

Conversation:
{conversation_text}

Respond with ONLY a decimal number between 0.0 and 1.0."""
        return await self._parse_score(prompt)

    async def _score_goal_completion(self, conversation_text: str, goal: str) -> float:
        """Score how well the conversation achieves the stated goal.

        Args:
            conversation_text: Formatted conversation transcript.
            goal: The intended objective of the conversation.

        Returns:
            Goal completion score in [0, 1].
        """
        prompt = f"""Rate goal completion on a scale of 0.0 to 1.0.
Goal: {goal}

Conversation:
{conversation_text}

Does the conversation fully accomplish the stated goal? Respond with ONLY a decimal number 0.0 to 1.0."""
        return await self._parse_score(prompt)

    async def _score_contradiction(self, conversation_text: str) -> float:
        """Estimate contradiction rate in the assistant's responses.

        Args:
            conversation_text: Formatted conversation transcript.

        Returns:
            Contradiction rate in [0, 1] (higher = more contradictions).
        """
        prompt = f"""Estimate the contradiction rate in the ASSISTANT's responses (0.0 = no contradictions, 1.0 = highly contradictory).

Conversation:
{conversation_text}

Respond with ONLY a decimal number between 0.0 and 1.0."""
        return await self._parse_score(prompt)

    async def _parse_score(self, prompt: str) -> float:
        """Ask the judge LLM and parse the float response.

        Args:
            prompt: Evaluation prompt for the judge.

        Returns:
            Parsed float score, clamped to [0, 1].
        """
        try:
            response = await self._llm_judge.complete(prompt)
            score = float(response.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning("conversation_evaluator_score_parse_failed", error=str(exc))
            return 0.5
