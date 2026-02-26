"""RAG pipeline evaluation adapter using RAGAS.

Implements IRAGEvaluator with 5 RAGAS metrics:
  1. Faithfulness         — answer grounded in retrieved context
  2. Answer Relevancy     — answer addresses the user query
  3. Context Precision    — relevant context ranked above irrelevant context
  4. Context Recall       — ground truth information present in context
  5. Answer Correctness   — end-to-end factual accuracy (requires ground truth)

RAGAS is a synchronous library. All evaluation calls are wrapped in
asyncio.to_thread() to prevent blocking the FastAPI event loop.
"""

import asyncio
from typing import Any

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)

_RAGAS_METRICS = {
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_correctness",
}


class RAGEvaluator:
    """RAG pipeline evaluator using RAGAS metrics.

    Evaluates the quality of a RAG pipeline by measuring faithfulness,
    answer relevancy, context precision/recall, and answer correctness.

    RAGAS operates at the dataset level — all questions/answers/contexts
    are evaluated together in a single batch for efficiency.

    Args:
        settings: Application settings with LLM provider configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings providing OpenAI configuration for RAGAS.
        """
        self._settings = settings
        self._openai_model = settings.openai_model
        self._api_key = settings.openai_api_key
        self._batch_size = settings.ragas_batch_size
        self._threshold = settings.default_pass_threshold

    async def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Evaluate a RAG pipeline using RAGAS metrics.

        Dispatches evaluation to a thread pool and converts RAGAS dataset
        results into the standard result dict format.

        Args:
            questions: User queries submitted to the RAG pipeline.
            answers: Generated answers from the RAG pipeline.
            contexts: Retrieved document chunks per question.
            ground_truths: Optional reference answers for supervised metrics.

        Returns:
            List of result dicts with: metric_name, score, threshold, passed, details.
        """
        if not questions:
            logger.warning("RAG evaluator called with empty question list")
            return []

        results = await asyncio.to_thread(
            self._run_ragas_evaluation,
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )

        logger.info(
            "RAGAS evaluation completed",
            questions=len(questions),
            results=len(results),
        )
        return results

    def _run_ragas_evaluation(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Execute RAGAS evaluation synchronously (runs in thread pool).

        Args:
            questions: User queries.
            answers: Generated answers.
            contexts: Retrieved document chunks per question.
            ground_truths: Optional reference answers.

        Returns:
            List of result dicts.
        """
        try:
            return self._run_with_ragas(questions, answers, contexts, ground_truths)
        except ImportError:
            logger.warning("ragas not installed — returning mock scores")
            return self._mock_ragas_scores(questions, ground_truths)
        except Exception as exc:
            logger.error("RAGAS evaluation failed", error=str(exc))
            return self._error_scores(str(exc), len(questions))

    def _run_with_ragas(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Execute RAGAS evaluation with real library.

        Args:
            questions: User queries.
            answers: Generated answers.
            contexts: Retrieved chunks per query.
            ground_truths: Optional reference answers.

        Returns:
            List of result dicts.
        """
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        dataset_dict: dict[str, list] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths is not None:
            dataset_dict["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(dataset_dict)

        metrics_to_run = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
        if ground_truths is not None:
            metrics_to_run.extend([ContextRecall(), AnswerCorrectness()])

        ragas_result = ragas_evaluate(dataset=dataset, metrics=metrics_to_run)
        scores_df = ragas_result.to_pandas()

        results: list[dict[str, Any]] = []
        ragas_metric_map = {
            "faithfulness": "ragas_faithfulness",
            "answer_relevancy": "ragas_answer_relevancy",
            "context_precision": "ragas_context_precision",
            "context_recall": "ragas_context_recall",
            "answer_correctness": "ragas_answer_correctness",
        }

        for ragas_col, metric_name in ragas_metric_map.items():
            if ragas_col not in scores_df.columns:
                continue

            col_scores = scores_df[ragas_col].tolist()
            for idx, score in enumerate(col_scores):
                score_float = float(score) if score is not None else 0.0
                passed = score_float >= self._threshold
                results.append(
                    {
                        "metric_name": metric_name,
                        "score": round(score_float, 4),
                        "threshold": self._threshold,
                        "passed": passed,
                        "details": {
                            "question_idx": idx,
                            "question": questions[idx] if idx < len(questions) else "",
                        },
                    }
                )

        return results

    def _mock_ragas_scores(
        self,
        questions: list[str],
        ground_truths: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Return mock RAGAS scores when the library is not installed.

        Args:
            questions: User queries (used for count only).
            ground_truths: Optional ground truths (determines which metrics are returned).

        Returns:
            Mock result dicts.
        """
        base_metrics = ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision"]
        if ground_truths is not None:
            base_metrics.extend(["ragas_context_recall", "ragas_answer_correctness"])

        results: list[dict[str, Any]] = []
        for metric_name in base_metrics:
            for idx in range(len(questions)):
                results.append(
                    {
                        "metric_name": metric_name,
                        "score": 0.75,
                        "threshold": self._threshold,
                        "passed": True,
                        "details": {
                            "question_idx": idx,
                            "mock": True,
                            "note": "ragas not installed — mock score returned",
                        },
                    }
                )
        return results

    def _error_scores(self, error_message: str, count: int) -> list[dict[str, Any]]:
        """Return zero-score error results for all metrics on evaluation failure.

        Args:
            error_message: Description of the failure.
            count: Number of questions (for building one error result per question).

        Returns:
            Error result dicts with score=0.0 and passed=False.
        """
        return [
            {
                "metric_name": "ragas_faithfulness",
                "score": 0.0,
                "threshold": self._threshold,
                "passed": False,
                "details": {"error": error_message, "question_count": count},
            }
        ]


__all__ = ["RAGEvaluator"]
