#!/usr/bin/env python3
"""
Run Benchmark Script
====================
Execute BLEU/BERTScore benchmark for RAG vs Baseline comparison.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.benchmark import TranslationBenchmark
from prompt_builder.augmented_prompt import AugmentedPromptBuilder
from config import BENCHMARK_TEST_SET


def mock_rag_translator(gloss: str) -> str:
    """
    Mock RAG translator for testing.
    In production, this would call the LLM with RAG-augmented prompt.
    """
    builder = AugmentedPromptBuilder()
    prompt = builder.build_prompt(gloss)
    
    # Return the gloss as lowercase (mock translation)
    # In production: call LLM with prompt
    words = gloss.lower().split()
    return " ".join(words) + "."


def mock_baseline_translator(gloss: str) -> str:
    """
    Mock baseline translator for testing.
    In production, this would call the LLM with simple prompt.
    """
    builder = AugmentedPromptBuilder()
    prompt = builder.build_simple_prompt(gloss)
    
    # Return simple mock translation
    words = gloss.lower().split()
    return " ".join(words) + "."


def main():
    print("=" * 60)
    print("TID RAG System - Benchmark Runner")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = TranslationBenchmark()
    
    # Load test set
    print(f"\nLoading test set from: {BENCHMARK_TEST_SET}")
    test_set = benchmark.load_test_set()
    
    if not test_set:
        print("No test set found. Creating sample test set...")
        test_set = [
            {"gloss": "BEN OKUL GITMEK", "reference": "Okula gidiyorum."},
            {"gloss": "SEN YEMEK YEMEK", "reference": "Yemek yiyorsun."},
            {"gloss": "AGAC UZUN YASAMAK", "reference": "Agac uzun yasar."},
        ]
    
    print(f"Test set size: {len(test_set)} samples")
    
    # Run benchmark
    print("\nRunning benchmark...")
    print("-" * 60)
    
    results = benchmark.run_benchmark(
        test_set,
        rag_translator=mock_rag_translator,
        baseline_translator=mock_baseline_translator,
    )
    
    # Print results
    print(benchmark.format_results(results))
    
    # Save results to JSON
    output_path = Path(__file__).parent.parent / "evaluation" / "benchmark_results.json"
    output_data = {
        method: {
            "bleu_score": result.bleu_score,
            "bertscore_precision": result.bertscore_precision,
            "bertscore_recall": result.bertscore_recall,
            "bertscore_f1": result.bertscore_f1,
            "num_samples": result.num_samples,
        }
        for method, result in results.items()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("NOTE: This benchmark uses MOCK translators.")
    print("For actual results, integrate with LLM services.")
    print("=" * 60)


if __name__ == "__main__":
    main()
