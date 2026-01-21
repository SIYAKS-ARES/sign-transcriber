"""
Experiment Runner for TID Translation System
=============================================
Core experiment logic for running translation benchmarks.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root (sign-transcriber/)
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)

from preprocessor import preprocess_text_for_llm, create_final_prompt, translate_with_rag, is_rag_available
from llm_services import translate_with_llm


@dataclass
class ExperimentResult:
    """Result of a single translation experiment."""
    gloss: str
    reference: str
    translation: str
    confidence: int
    alternatives: List[Dict] = field(default_factory=list)
    explanation: str = ""
    error: Optional[str] = None
    rag_used: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentBatchResult:
    """Result of a batch of experiments."""
    word_count: int
    total_samples: int
    successful: int
    failed: int
    avg_confidence: float
    avg_latency_ms: float
    results: List[ExperimentResult] = field(default_factory=list)
    timestamp: str = ""
    provider: str = "gemini"
    rag_available: bool = False
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "word_count": self.word_count,
            "total_samples": self.total_samples,
            "successful": self.successful,
            "failed": self.failed,
            "avg_confidence": self.avg_confidence,
            "avg_latency_ms": self.avg_latency_ms,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "rag_available": self.rag_available,
            "results": [r.to_dict() for r in self.results],
        }


class ExperimentRunner:
    """
    Runs translation experiments on test sets.
    
    Supports:
    - Running experiments by word count (3, 4, 5)
    - Using RAG pipeline or direct LLM
    - Collecting metrics (confidence, latency, success rate)
    """
    
    TEST_SETS_DIR = Path(__file__).parent / "test_sets"
    
    def __init__(self, provider: str = "gemini", use_rag: bool = True):
        """
        Initialize the experiment runner.
        
        Args:
            provider: LLM provider ("gemini", "openai", "claude")
            use_rag: Whether to use RAG pipeline
        """
        self.provider = provider
        self.use_rag = use_rag
        self.rag_available = is_rag_available() if use_rag else False
    
    def load_test_set(self, word_count: int) -> List[Dict]:
        """Load test set for given word count."""
        test_file = self.TEST_SETS_DIR / f"{word_count}_word_glosses.json"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test set not found: {test_file}")
        
        with open(test_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def run_single_experiment(self, gloss: str, reference: str) -> ExperimentResult:
        """
        Run a single translation experiment.
        
        Args:
            gloss: Input TID gloss
            reference: Reference translation
            
        Returns:
            ExperimentResult with translation and metrics
        """
        start_time = time.time()
        
        try:
            if self.use_rag and self.rag_available:
                # Use full RAG pipeline
                result = translate_with_rag(gloss)
                translation = result.get("translation", "")
                confidence = result.get("confidence", 0)
                alternatives = result.get("alternatives", [])
                explanation = result.get("explanation", "")
                error = result.get("error")
                rag_used = result.get("rag_used", False)
            else:
                # Use direct LLM with preprocessor
                processed = preprocess_text_for_llm(gloss)
                prompt = create_final_prompt(processed)
                result = translate_with_llm(self.provider, prompt)
                translation = result.get("translation", "")
                confidence = result.get("confidence", 0)
                alternatives = result.get("alternatives", [])
                explanation = result.get("explanation", "")
                error = result.get("error")
                rag_used = False
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ExperimentResult(
                gloss=gloss,
                reference=reference,
                translation=translation,
                confidence=confidence,
                alternatives=alternatives,
                explanation=explanation,
                error=error,
                rag_used=rag_used,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                gloss=gloss,
                reference=reference,
                translation="",
                confidence=0,
                error=str(e),
                latency_ms=latency_ms,
            )
    
    def run_batch(
        self, 
        word_count: int, 
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> ExperimentBatchResult:
        """
        Run experiments on a batch of test samples.
        
        Args:
            word_count: Word count (3, 4, or 5)
            limit: Optional limit on number of samples
            verbose: Print progress
            
        Returns:
            ExperimentBatchResult with all results
        """
        test_data = self.load_test_set(word_count)
        
        if limit:
            test_data = test_data[:limit]
        
        results = []
        successful = 0
        total_confidence = 0
        total_latency = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {word_count}-word experiments ({len(test_data)} samples)")
            print(f"Provider: {self.provider}, RAG: {self.rag_available}")
            print(f"{'='*60}")
        
        for i, sample in enumerate(test_data):
            gloss = sample["gloss"]
            reference = sample["reference"]
            
            if verbose:
                print(f"\n[{i+1}/{len(test_data)}] {gloss}")
            
            result = self.run_single_experiment(gloss, reference)
            results.append(result)
            
            if result.translation and not result.error:
                successful += 1
                total_confidence += result.confidence
            
            total_latency += result.latency_ms
            
            if verbose:
                status = "OK" if not result.error else f"ERROR: {result.error}"
                print(f"  -> {result.translation[:50]}... [{result.confidence}/10] ({status})")
        
        avg_confidence = total_confidence / successful if successful > 0 else 0
        avg_latency = total_latency / len(test_data) if test_data else 0
        
        batch_result = ExperimentBatchResult(
            word_count=word_count,
            total_samples=len(test_data),
            successful=successful,
            failed=len(test_data) - successful,
            avg_confidence=round(avg_confidence, 2),
            avg_latency_ms=round(avg_latency, 2),
            results=results,
            provider=self.provider,
            rag_available=self.rag_available,
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Results: {successful}/{len(test_data)} successful")
            print(f"Avg Confidence: {batch_result.avg_confidence}/10")
            print(f"Avg Latency: {batch_result.avg_latency_ms:.0f}ms")
            print(f"{'='*60}")
        
        return batch_result
    
    def run_all(
        self, 
        word_counts: List[int] = [3, 4, 5],
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[int, ExperimentBatchResult]:
        """
        Run experiments on all word counts.
        
        Args:
            word_counts: List of word counts to test
            limit: Optional limit per word count
            verbose: Print progress
            
        Returns:
            Dictionary mapping word count to batch results
        """
        results = {}
        
        for word_count in word_counts:
            try:
                batch_result = self.run_batch(word_count, limit=limit, verbose=verbose)
                results[word_count] = batch_result
            except Exception as e:
                print(f"Error running {word_count}-word experiments: {e}")
        
        return results


if __name__ == "__main__":
    # Quick test
    runner = ExperimentRunner(provider="gemini", use_rag=True)
    
    print("Testing single experiment...")
    result = runner.run_single_experiment(
        gloss="BEN OKUL GITMEK",
        reference="Okula gidiyorum."
    )
    print(f"Result: {result.translation} [{result.confidence}/10]")
    print(f"RAG used: {result.rag_used}")
    print(f"Latency: {result.latency_ms:.0f}ms")
