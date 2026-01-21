#!/usr/bin/env python3
"""
End-to-End Test Script for TID RAG System
==========================================
Tests all components of the RAG system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test configuration loading."""
    print("\n[1/7] Testing Configuration...")
    from config import (
        EMBEDDING_MODEL,
        DISTANCE_METRIC,
        SOZLUK_COLLECTION,
        HAFIZA_COLLECTION,
        VECTORSTORE_PATH,
    )
    
    assert EMBEDDING_MODEL is not None, "EMBEDDING_MODEL not set"
    assert DISTANCE_METRIC == "cosine", f"Unexpected distance metric: {DISTANCE_METRIC}"
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - Distance metric: {DISTANCE_METRIC}")
    print(f"  - Collections: {SOZLUK_COLLECTION}, {HAFIZA_COLLECTION}")
    print("  [PASS] Configuration OK")


def test_preprocessing():
    """Test preprocessing functions."""
    print("\n[2/7] Testing Preprocessing...")
    from preprocessing.cleaning import (
        clean_sozluk_entry,
        normalize_gloss,
        remove_boilerplate,
    )
    
    # Test normalize_gloss
    result = normalize_gloss("okul gitmek istemek")
    assert result == "OKUL GITMEK ISTEMEK", f"Unexpected result: {result}"
    
    # Test remove_boilerplate
    text = "Guncel Turk Isaret Dili Sozlugu\nTest content"
    result = remove_boilerplate(text)
    assert "Test content" in result, f"Boilerplate not removed: {result}"
    
    print("  - normalize_gloss: OK")
    print("  - remove_boilerplate: OK")
    print("  [PASS] Preprocessing OK")


def test_collections():
    """Test ChromaDB collections."""
    print("\n[3/7] Testing Collections...")
    from tid_collections.sozluk_collection import SozlukCollection
    from tid_collections.hafiza_collection import HafizaCollection
    
    sozluk = SozlukCollection()
    hafiza = HafizaCollection()
    
    sozluk_count = sozluk.get_count()
    hafiza_count = hafiza.get_count()
    
    print(f"  - Sozluk documents: {sozluk_count}")
    print(f"  - Hafiza documents: {hafiza_count}")
    
    assert sozluk_count > 0, "Sozluk collection is empty"
    assert hafiza_count > 0, "Hafiza collection is empty"
    
    print("  [PASS] Collections OK")


def test_retrieval():
    """Test dual retrieval."""
    print("\n[4/7] Testing Dual Retrieval...")
    from retriever.dual_retriever import DualRetriever
    
    retriever = DualRetriever()
    
    # Test retrieval
    result = retriever.retrieve("OKUL GITMEK ISTEMEK")
    
    print(f"  - Similar translations found: {len(result.similar_translations)}")
    print(f"  - Words with info: {len([w for w, i in result.word_info.items() if i])}")
    
    assert result.query == "OKUL GITMEK ISTEMEK", "Query not preserved"
    
    # Test context string generation
    context = result.to_context_string()
    assert len(context) > 0, "Empty context string"
    print(f"  - Context string length: {len(context)} chars")
    
    print("  [PASS] Dual Retrieval OK")


def test_prompt_builder():
    """Test prompt building."""
    print("\n[5/7] Testing Prompt Builder...")
    from prompt_builder.augmented_prompt import AugmentedPromptBuilder
    
    builder = AugmentedPromptBuilder()
    
    # Test augmented prompt
    prompt = builder.build_prompt("OKUL GITMEK")
    assert "OKUL GITMEK" in prompt, "Transcription not in prompt"
    assert "RAG" in prompt or "REFERANS" in prompt, "RAG context missing"
    print(f"  - Augmented prompt length: {len(prompt)} chars")
    
    # Test simple prompt
    simple = builder.build_simple_prompt("OKUL GITMEK")
    assert len(simple) < len(prompt), "Simple prompt should be shorter"
    print(f"  - Simple prompt length: {len(simple)} chars")
    
    print("  [PASS] Prompt Builder OK")


def test_input_adapter():
    """Test input adapter."""
    print("\n[6/7] Testing Input Adapter...")
    from integration.input_adapter import adapt_input, TranscriptionInput
    
    # Test string input
    result = adapt_input("okul gitmek")
    assert result.raw_string == "OKUL GITMEK", f"Unexpected: {result.raw_string}"
    
    # Test list input
    result = adapt_input(["OKUL", "GITMEK"])
    assert result.glosses == ["OKUL", "GITMEK"], f"Unexpected: {result.glosses}"
    
    # Test dict input
    result = adapt_input({"glosses": ["okul"], "confidence": 0.9})
    assert result.confidence == 0.9, f"Unexpected confidence: {result.confidence}"
    
    print("  - String input: OK")
    print("  - List input: OK")
    print("  - Dict input: OK")
    print("  [PASS] Input Adapter OK")


def test_feedback_handler():
    """Test feedback handler."""
    print("\n[7/7] Testing Feedback Handler...")
    from feedback.feedback_handler import FeedbackHandler
    
    handler = FeedbackHandler()
    
    initial_count = handler.get_hafiza_stats()["total_translations"]
    
    # Create feedback
    feedback_id = handler.create_feedback(
        transkripsiyon="TEST FEEDBACK",
        generated_translation="Test cevirisi",
        provider="test",
        confidence=0.8,
    )
    
    assert feedback_id is not None, "Feedback ID not generated"
    print(f"  - Created feedback: {feedback_id}")
    
    # Check pending
    pending = handler.get_pending()
    assert len(pending) > 0, "No pending feedback"
    
    # Reject it (don't save to hafiza)
    handler.reject_translation(feedback_id, "Test rejection")
    
    final_count = handler.get_hafiza_stats()["total_translations"]
    assert final_count == initial_count, "Rejected feedback was saved"
    
    print("  - Feedback creation: OK")
    print("  - Rejection handling: OK")
    print("  [PASS] Feedback Handler OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TID RAG System - End-to-End Tests")
    print("=" * 60)
    
    tests = [
        test_config,
        test_preprocessing,
        test_collections,
        test_retrieval,
        test_prompt_builder,
        test_input_adapter,
        test_feedback_handler,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
