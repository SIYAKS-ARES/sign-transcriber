"""
Benchmark Module for Academic Evaluation
=========================================
Calculates BLEU and BERTScore for RAG vs Baseline comparison.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARK_TEST_SET, ENABLE_BASELINE_COMPARISON


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method: str  # "rag" or "baseline"
    bleu_score: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    num_samples: int = 0
    translations: List[Dict] = field(default_factory=list)


class TranslationBenchmark:
    """
    Benchmark module for comparing RAG vs Baseline translation quality.
    
    Metrics:
    - BLEU Score: N-gram overlap (standard MT metric)
    - BERTScore: Semantic similarity (important for sign language)
    """
    
    def __init__(self):
        """Initialize the benchmark with metric calculators."""
        self._bleu = None
        self._bert_score_available = False
        
        # Lazy load metrics
        try:
            from sacrebleu.metrics import BLEU
            self._bleu = BLEU()
        except ImportError:
            print("Warning: sacrebleu not installed. BLEU scores will be 0.")
        
        try:
            import bert_score
            self._bert_score_available = True
        except ImportError:
            print("Warning: bert-score not installed. BERTScore will be 0.")
    
    def load_test_set(self, path: Optional[Path] = None) -> List[Dict]:
        """
        Load test set from JSON file.
        
        Expected format:
        [
            {"gloss": "OKUL GITMEK", "reference": "Okula gidiyorum."},
            ...
        ]
        """
        path = path or BENCHMARK_TEST_SET
        
        if not Path(path).exists():
            print(f"Test set not found at {path}. Returning empty list.")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_bleu(
        self, 
        hypotheses: List[str], 
        references: List[str]
    ) -> float:
        """Calculate corpus-level BLEU score."""
        if not self._bleu or not hypotheses or not references:
            return 0.0
        
        try:
            result = self._bleu.corpus_score(hypotheses, [references])
            return result.score
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_bertscore(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate BERTScore metrics."""
        if not self._bert_score_available or not hypotheses or not references:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            from bert_score import score
            P, R, F1 = score(hypotheses, references, lang="tr", verbose=False)
            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item(),
            }
        except Exception as e:
            print(f"BERTScore calculation error: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def run_benchmark(
        self,
        test_set: List[Dict],
        rag_translator: callable,
        baseline_translator: Optional[callable] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark on test set.
        
        Args:
            test_set: List of {"gloss": str, "reference": str}
            rag_translator: Function (gloss) -> translation (with RAG)
            baseline_translator: Function (gloss) -> translation (without RAG)
            
        Returns:
            Dictionary with "rag" and optionally "baseline" results
        """
        results = {}
        
        # Collect translations
        rag_hypotheses = []
        baseline_hypotheses = []
        references = []
        
        for item in test_set:
            gloss = item["gloss"]
            reference = item["reference"]
            references.append(reference)
            
            # RAG translation
            rag_trans = rag_translator(gloss)
            rag_hypotheses.append(rag_trans)
            
            # Baseline translation
            if baseline_translator and ENABLE_BASELINE_COMPARISON:
                baseline_trans = baseline_translator(gloss)
                baseline_hypotheses.append(baseline_trans)
        
        # Calculate RAG metrics
        rag_bleu = self.calculate_bleu(rag_hypotheses, references)
        rag_bert = self.calculate_bertscore(rag_hypotheses, references)
        
        results["rag"] = BenchmarkResult(
            method="rag",
            bleu_score=rag_bleu,
            bertscore_precision=rag_bert["precision"],
            bertscore_recall=rag_bert["recall"],
            bertscore_f1=rag_bert["f1"],
            num_samples=len(test_set),
            translations=[
                {"gloss": t["gloss"], "hypothesis": h, "reference": t["reference"]}
                for t, h in zip(test_set, rag_hypotheses)
            ],
        )
        
        # Calculate baseline metrics if available
        if baseline_hypotheses:
            baseline_bleu = self.calculate_bleu(baseline_hypotheses, references)
            baseline_bert = self.calculate_bertscore(baseline_hypotheses, references)
            
            results["baseline"] = BenchmarkResult(
                method="baseline",
                bleu_score=baseline_bleu,
                bertscore_precision=baseline_bert["precision"],
                bertscore_recall=baseline_bert["recall"],
                bertscore_f1=baseline_bert["f1"],
                num_samples=len(test_set),
                translations=[
                    {"gloss": t["gloss"], "hypothesis": h, "reference": t["reference"]}
                    for t, h in zip(test_set, baseline_hypotheses)
                ],
            )
        
        return results
    
    def format_results(self, results: Dict[str, BenchmarkResult]) -> str:
        """Format benchmark results as a readable string."""
        lines = ["=" * 60, "BENCHMARK RESULTS", "=" * 60]
        
        for method, result in results.items():
            lines.append(f"\n{method.upper()} Method:")
            lines.append(f"  Samples: {result.num_samples}")
            lines.append(f"  BLEU Score: {result.bleu_score:.2f}")
            lines.append(f"  BERTScore:")
            lines.append(f"    Precision: {result.bertscore_precision:.4f}")
            lines.append(f"    Recall: {result.bertscore_recall:.4f}")
            lines.append(f"    F1: {result.bertscore_f1:.4f}")
        
        if "rag" in results and "baseline" in results:
            lines.append("\n" + "-" * 60)
            lines.append("COMPARISON (RAG vs Baseline):")
            diff_bleu = results["rag"].bleu_score - results["baseline"].bleu_score
            diff_f1 = results["rag"].bertscore_f1 - results["baseline"].bertscore_f1
            lines.append(f"  BLEU Improvement: {diff_bleu:+.2f}")
            lines.append(f"  BERTScore F1 Improvement: {diff_f1:+.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class BaselineTranslator:
    """Simple baseline translator for comparison (zero-shot LLM)."""
    
    def __init__(self, llm_provider: str = "gemini"):
        """Initialize baseline translator."""
        self.provider = llm_provider
    
    def translate(self, gloss: str) -> str:
        """
        Translate using simple zero-shot prompt.
        
        Note: This is a placeholder. Actual implementation would
        call the LLM service.
        """
        # This would call llm_services.translate_with_llm
        # For now, return a placeholder
        return f"[Baseline translation of: {gloss}]"


if __name__ == "__main__":
    # Test the benchmark module
    print("Testing Benchmark Module...")
    print("=" * 60)
    
    benchmark = TranslationBenchmark()
    
    # Create a small test set
    test_set = [
        {"gloss": "BEN OKUL GITMEK", "reference": "Okula gidiyorum."},
        {"gloss": "SEN YEMEK YEMEK", "reference": "Yemek yiyorsun."},
    ]
    
    # Mock translators for testing
    def mock_rag_translator(gloss: str) -> str:
        return f"RAG cevirisi: {gloss.lower()}"
    
    def mock_baseline_translator(gloss: str) -> str:
        return f"Baseline: {gloss.lower()}"
    
    results = benchmark.run_benchmark(
        test_set,
        mock_rag_translator,
        mock_baseline_translator,
    )
    
    print(benchmark.format_results(results))
