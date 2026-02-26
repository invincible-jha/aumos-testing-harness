"""LLM evaluation adapter using deepeval.

Implements ILLMEvaluator with 14 evaluation metrics:
  1.  Accuracy          — factual correctness against ground truth (GEval)
  2.  Coherence         — logical flow and consistency (GEval)
  3.  Faithfulness      — output grounded in context
  4.  Answer Relevancy  — relevance to the question
  5.  Contextual Precision  — relevant context ranked higher
  6.  Contextual Recall     — all relevant context retrieved
  7.  Contextual Relevancy  — context matched to query
  8.  Hallucination Detection — unsupported claims
  9.  Toxicity          — harmful or offensive content
  10. Bias Detection    — demographic or ideological bias
  11. Summarization Quality — fidelity and coverage
  12. Task Completion   — goal achievement (GEval)
  13. Tool Call Accuracy — correct tool and argument selection
  14. Latency Score     — response time relative to SLA threshold

All deepeval metric calls are executed via asyncio.to_thread() because the
deepeval library's sync API is not natively async. Never block the event loop.
"""

import asyncio
import time
from typing import Any

from aumos_common.observability import get_logger

from aumos_testing_harness.settings import Settings

logger = get_logger(__name__)

# Metric names as string constants matching MetricName enum values
_LLM_METRIC_MAP: dict[str, str] = {
    "accuracy": "accuracy",
    "coherence": "coherence",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "contextual_precision": "contextual_precision",
    "contextual_recall": "contextual_recall",
    "contextual_relevancy": "contextual_relevancy",
    "hallucination": "hallucination",
    "toxicity": "toxicity",
    "bias": "bias",
    "summarization": "summarization",
    "task_completion": "task_completion",
    "tool_call_accuracy": "tool_call_accuracy",
    "latency_score": "latency_score",
}


class LLMEvaluator:
    """LLM quality evaluator using the deepeval framework.

    Provides 14 metric scorers against test case inputs and outputs.
    All synchronous deepeval calls are wrapped in asyncio.to_thread()
    to avoid blocking the FastAPI event loop.

    Args:
        settings: Application settings with OpenAI API key and model configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings.

        Args:
            settings: Application settings providing LLM provider configuration.
        """
        self._settings = settings
        self._openai_model = settings.openai_model
        self._api_key = settings.openai_api_key

    async def evaluate(
        self,
        metric_names: list[str],
        test_cases: list[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Run one or more LLM metrics against a batch of test cases.

        Each (metric, test_case) pair produces one result dict. The evaluation
        is dispatched to a thread pool to prevent blocking the event loop.

        Args:
            metric_names: List of metric names from MetricName enum.
            test_cases: List of test case dicts. Each must have 'input' and
                'actual_output'. Optional fields: 'expected_output', 'context',
                'retrieval_context', 'latency_ms'.
            threshold: Pass/fail threshold (0.0-1.0).

        Returns:
            List of result dicts with: metric_name, score, threshold, passed, details.
        """
        results: list[dict[str, Any]] = []

        for metric_name in metric_names:
            if metric_name not in _LLM_METRIC_MAP:
                logger.warning("Unknown LLM metric skipped", metric_name=metric_name)
                continue

            for idx, test_case in enumerate(test_cases):
                result = await asyncio.to_thread(
                    self._score_metric,
                    metric_name=metric_name,
                    test_case=test_case,
                    threshold=threshold,
                    test_case_idx=idx,
                )
                results.append(result)

        logger.info(
            "LLM evaluation completed",
            metrics=metric_names,
            test_cases=len(test_cases),
            results=len(results),
        )
        return results

    def _score_metric(
        self,
        metric_name: str,
        test_case: dict[str, Any],
        threshold: float,
        test_case_idx: int,
    ) -> dict[str, Any]:
        """Score a single metric for a single test case (synchronous).

        This method runs in a thread pool. It imports deepeval lazily to
        avoid startup overhead when evaluators are not in use.

        Args:
            metric_name: Name of the metric to compute.
            test_case: Test case data dict.
            threshold: Pass/fail threshold.
            test_case_idx: Test case index for tracing.

        Returns:
            Result dict with: metric_name, score, threshold, passed, details.
        """
        try:
            return self._run_deepeval_metric(metric_name, test_case, threshold, test_case_idx)
        except ImportError:
            logger.warning("deepeval not installed — returning mock score", metric_name=metric_name)
            return self._mock_score(metric_name, test_case, threshold, test_case_idx)
        except Exception as exc:
            logger.error(
                "Metric scoring failed",
                metric_name=metric_name,
                test_case_idx=test_case_idx,
                error=str(exc),
            )
            return {
                "metric_name": metric_name,
                "score": 0.0,
                "threshold": threshold,
                "passed": False,
                "details": {
                    "test_case_idx": test_case_idx,
                    "error": str(exc),
                    "input": test_case.get("input", ""),
                },
            }

    def _run_deepeval_metric(
        self,
        metric_name: str,
        test_case: dict[str, Any],
        threshold: float,
        test_case_idx: int,
    ) -> dict[str, Any]:
        """Execute deepeval metric scoring.

        Args:
            metric_name: Metric identifier.
            test_case: Test case data.
            threshold: Pass/fail threshold.
            test_case_idx: Test case index for tracing.

        Returns:
            Scored result dict.
        """
        from deepeval import evaluate as deepeval_evaluate  # noqa: PLC0415
        from deepeval.metrics import (  # noqa: PLC0415
            AnswerRelevancyMetric,
            BiasMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            GEval,
            HallucinationMetric,
            SummarizationMetric,
            ToxicityMetric,
        )
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: PLC0415

        lm_test_case = LLMTestCase(
            input=test_case.get("input", ""),
            actual_output=test_case.get("actual_output", ""),
            expected_output=test_case.get("expected_output"),
            context=test_case.get("context"),
            retrieval_context=test_case.get("retrieval_context"),
        )

        metric_instance: Any
        start_time = time.monotonic()

        if metric_name == "accuracy":
            metric_instance = GEval(
                name="Accuracy",
                criteria="Determine whether the actual output is factually correct based on the expected output.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                threshold=threshold,
                model=self._openai_model,
            )
        elif metric_name == "coherence":
            metric_instance = GEval(
                name="Coherence",
                criteria="Evaluate whether the actual output is logically coherent and internally consistent.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=threshold,
                model=self._openai_model,
            )
        elif metric_name == "faithfulness":
            metric_instance = FaithfulnessMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "answer_relevancy":
            metric_instance = AnswerRelevancyMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "contextual_precision":
            metric_instance = ContextualPrecisionMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "contextual_recall":
            metric_instance = ContextualRecallMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "contextual_relevancy":
            metric_instance = ContextualRelevancyMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "hallucination":
            metric_instance = HallucinationMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "toxicity":
            metric_instance = ToxicityMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "bias":
            metric_instance = BiasMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "summarization":
            metric_instance = SummarizationMetric(threshold=threshold, model=self._openai_model)
        elif metric_name == "task_completion":
            metric_instance = GEval(
                name="TaskCompletion",
                criteria="Evaluate whether the actual output successfully completes the task described in the input.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=threshold,
                model=self._openai_model,
            )
        elif metric_name == "tool_call_accuracy":
            metric_instance = GEval(
                name="ToolCallAccuracy",
                criteria=(
                    "Evaluate whether the tool calls in the actual output use the correct tool name "
                    "and provide accurate arguments as specified in the expected output."
                ),
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                threshold=threshold,
                model=self._openai_model,
            )
        elif metric_name == "latency_score":
            latency_ms = test_case.get("latency_ms", 0)
            sla_ms = test_case.get("sla_ms", 2000)
            score = max(0.0, min(1.0, 1.0 - (latency_ms - sla_ms) / max(sla_ms, 1)))
            passed = score >= threshold
            return {
                "metric_name": metric_name,
                "score": round(score, 4),
                "threshold": threshold,
                "passed": passed,
                "details": {
                    "test_case_idx": test_case_idx,
                    "latency_ms": latency_ms,
                    "sla_ms": sla_ms,
                },
            }
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

        metric_instance.measure(lm_test_case)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        score = float(metric_instance.score)
        passed = score >= threshold

        return {
            "metric_name": metric_name,
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
            "details": {
                "test_case_idx": test_case_idx,
                "input": test_case.get("input", ""),
                "actual_output": test_case.get("actual_output", ""),
                "reason": getattr(metric_instance, "reason", ""),
                "eval_duration_ms": round(elapsed_ms, 1),
            },
        }

    def _mock_score(
        self,
        metric_name: str,
        test_case: dict[str, Any],
        threshold: float,
        test_case_idx: int,
    ) -> dict[str, Any]:
        """Return a neutral mock score when deepeval is not installed.

        Used for local development without LLM API keys configured.

        Args:
            metric_name: Metric identifier.
            test_case: Test case data.
            threshold: Pass/fail threshold.
            test_case_idx: Test case index.

        Returns:
            Mock result dict with score=0.75.
        """
        mock_score = 0.75
        return {
            "metric_name": metric_name,
            "score": mock_score,
            "threshold": threshold,
            "passed": mock_score >= threshold,
            "details": {
                "test_case_idx": test_case_idx,
                "mock": True,
                "note": "deepeval not installed — mock score returned",
                "input": test_case.get("input", ""),
            },
        }


__all__ = ["LLMEvaluator"]
