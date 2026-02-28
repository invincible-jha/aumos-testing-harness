"""Benchmark dataset library for aumos-testing-harness.

Provides a curated registry of standard AI evaluation benchmarks
with access methods for downloading and using them in evaluation runs.

GAP-183: Benchmark Dataset Library
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkMetadata:
    """Metadata for a single benchmark dataset.

    Attributes:
        name: Short identifier for the benchmark.
        display_name: Human-readable benchmark name.
        description: Brief description of what the benchmark measures.
        task_type: Evaluation task type (e.g., 'qa', 'reasoning', 'coding').
        n_samples: Total number of evaluation samples.
        languages: Language codes covered by the benchmark.
        license: Data license identifier.
        citation: BibTeX citation or reference URL.
        s3_uri: S3/MinIO URI where the benchmark dataset is stored.
    """

    name: str
    display_name: str
    description: str
    task_type: str
    n_samples: int
    languages: list[str]
    license: str
    citation: str
    s3_uri: str


# Curated benchmark registry
BENCHMARK_REGISTRY: dict[str, BenchmarkMetadata] = {
    "mmlu": BenchmarkMetadata(
        name="mmlu",
        display_name="Massive Multitask Language Understanding (MMLU)",
        description="57-subject academic knowledge evaluation covering STEM, humanities, and social sciences",
        task_type="multiple_choice_qa",
        n_samples=14042,
        languages=["en"],
        license="MIT",
        citation="Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding.",
        s3_uri="s3://aumos-benchmarks/mmlu/mmlu_test.jsonl",
    ),
    "hellaswag": BenchmarkMetadata(
        name="hellaswag",
        display_name="HellaSwag",
        description="Commonsense NLI for grounding physical situational reasoning",
        task_type="multiple_choice_completion",
        n_samples=10042,
        languages=["en"],
        license="MIT",
        citation="Zellers et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?",
        s3_uri="s3://aumos-benchmarks/hellaswag/hellaswag_val.jsonl",
    ),
    "truthfulqa": BenchmarkMetadata(
        name="truthfulqa",
        display_name="TruthfulQA",
        description="Measures whether LLMs generate truthful answers, especially on misconception-prone topics",
        task_type="truthfulness",
        n_samples=817,
        languages=["en"],
        license="Apache-2.0",
        citation="Lin et al. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods.",
        s3_uri="s3://aumos-benchmarks/truthfulqa/truthfulqa.jsonl",
    ),
    "gsm8k": BenchmarkMetadata(
        name="gsm8k",
        display_name="GSM8K (Grade School Math)",
        description="8.5K grade school math word problems requiring multi-step arithmetic reasoning",
        task_type="math_reasoning",
        n_samples=1319,
        languages=["en"],
        license="MIT",
        citation="Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.",
        s3_uri="s3://aumos-benchmarks/gsm8k/gsm8k_test.jsonl",
    ),
    "humaneval": BenchmarkMetadata(
        name="humaneval",
        display_name="HumanEval",
        description="164 hand-crafted Python programming problems for measuring code generation capability",
        task_type="code_generation",
        n_samples=164,
        languages=["en"],
        license="MIT",
        citation="Chen et al. (2021). Evaluating Large Language Models Trained on Code.",
        s3_uri="s3://aumos-benchmarks/humaneval/humaneval.jsonl",
    ),
    "ragas_eval": BenchmarkMetadata(
        name="ragas_eval",
        display_name="AumOS RAG Evaluation Set",
        description="Curated enterprise RAG evaluation set covering customer service, legal, and technical domains",
        task_type="rag_qa",
        n_samples=2500,
        languages=["en"],
        license="Apache-2.0",
        citation="AumOS Enterprise internal benchmark (2024)",
        s3_uri="s3://aumos-benchmarks/ragas_eval/aumos_rag_eval.jsonl",
    ),
    "bbq": BenchmarkMetadata(
        name="bbq",
        display_name="BBQ (Bias Benchmark for QA)",
        description="Measures social biases in LLM outputs across 9 social dimensions",
        task_type="bias_detection",
        n_samples=58492,
        languages=["en"],
        license="CC-BY-4.0",
        citation="Parrish et al. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering.",
        s3_uri="s3://aumos-benchmarks/bbq/bbq.jsonl",
    ),
    "arc_challenge": BenchmarkMetadata(
        name="arc_challenge",
        display_name="ARC Challenge",
        description="Grade-school science questions requiring reasoning beyond simple retrieval",
        task_type="multiple_choice_qa",
        n_samples=1172,
        languages=["en"],
        license="CC-BY-SA-4.0",
        citation="Clark et al. (2018). Think You Have Solved Question Answering?",
        s3_uri="s3://aumos-benchmarks/arc/arc_challenge_test.jsonl",
    ),
}


class BenchmarkLibrary:
    """Provides access to the curated AumOS benchmark dataset registry.

    Args:
        s3_client: Optional S3/MinIO client for downloading benchmark data.
            If None, metadata-only operations are available.
    """

    def __init__(self, s3_client: Any | None = None) -> None:
        """Initialise the benchmark library.

        Args:
            s3_client: S3 client for data download operations.
        """
        self._s3_client = s3_client
        self._registry = BENCHMARK_REGISTRY

    def list_benchmarks(self, task_type: str | None = None) -> list[dict[str, Any]]:
        """List available benchmarks, optionally filtered by task type.

        Args:
            task_type: Filter by evaluation task type (e.g., 'qa', 'reasoning').

        Returns:
            List of benchmark metadata dictionaries.
        """
        benchmarks = list(self._registry.values())
        if task_type:
            benchmarks = [b for b in benchmarks if b.task_type == task_type]
        return [self._to_dict(b) for b in benchmarks]

    def get_metadata(self, benchmark_name: str) -> dict[str, Any]:
        """Get metadata for a specific benchmark.

        Args:
            benchmark_name: Short benchmark identifier.

        Returns:
            Benchmark metadata dictionary.

        Raises:
            KeyError: If the benchmark is not in the registry.
        """
        if benchmark_name not in self._registry:
            raise KeyError(f"Benchmark '{benchmark_name}' not found in registry")
        return self._to_dict(self._registry[benchmark_name])

    async def load_samples(
        self,
        benchmark_name: str,
        n_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load benchmark samples from S3/MinIO storage.

        Args:
            benchmark_name: Short benchmark identifier.
            n_samples: Maximum samples to load. None loads all.

        Returns:
            List of sample dicts with 'input', 'expected_output', and 'metadata'.

        Raises:
            KeyError: If the benchmark is not in the registry.
            RuntimeError: If S3 client is not configured.
        """
        if self._s3_client is None:
            raise RuntimeError("S3 client required for loading benchmark data")

        benchmark = self._registry.get(benchmark_name)
        if benchmark is None:
            raise KeyError(f"Benchmark '{benchmark_name}' not found")

        logger.info("benchmark_loading", name=benchmark_name, s3_uri=benchmark.s3_uri)
        try:
            samples = await self._s3_client.read_jsonl(benchmark.s3_uri, limit=n_samples)
            logger.info("benchmark_loaded", name=benchmark_name, n_samples=len(samples))
            return samples
        except Exception as exc:
            logger.error("benchmark_load_failed", name=benchmark_name, error=str(exc))
            raise

    def _to_dict(self, metadata: BenchmarkMetadata) -> dict[str, Any]:
        """Convert BenchmarkMetadata to a plain dictionary.

        Args:
            metadata: Benchmark metadata dataclass.

        Returns:
            Dictionary representation.
        """
        return {
            "name": metadata.name,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "task_type": metadata.task_type,
            "n_samples": metadata.n_samples,
            "languages": metadata.languages,
            "license": metadata.license,
            "citation": metadata.citation,
            "s3_uri": metadata.s3_uri,
        }
