#!/usr/bin/env python3
"""
TID Linguistic Feature Tests
=============================
Tests for TID preprocessor, few-shot builder, and response parser.
No API key required - tests preprocessing and parsing only.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.tid_preprocessor import TIDPreprocessor, PreprocessedInput
from prompt_builder.few_shot_builder import FewShotBuilder
from prompt_builder.templates import build_user_prompt, build_linguistic_context
from llm.response_parser import ResponseParser, TranslationResult


def test_preprocessor():
    """Test TID Preprocessor linguistic feature detection."""
    print("\n" + "=" * 60)
    print("TEST: TIDPreprocessor")
    print("=" * 60)
    
    preprocessor = TIDPreprocessor()
    
    test_cases = [
        # (input, expected_tense, expected_question, expected_negative, expected_repetitions)
        ("DÜN OKUL GITMEK", "past", False, False, {}),
        ("YARIN TOPLANTI VAR", "future", False, False, {}),
        ("SIMDI YEMEK YEMEK", "present", False, False, {"YEMEK": 2}),  # Repetition detected
        ("YEMEK YEMEK BITMEK", "past", False, False, {"YEMEK": 2}),  # BITMEK -> past, repetition
        ("SEN NEREDE CALISMAK", None, True, False, {}),  # Question
        ("BEN GITMEK DEGIL", None, False, True, {}),  # Negation
        ("GEZMEK GEZMEK YORULMAK", None, False, False, {"GEZMEK": 2}),  # Repetition
        ("YÜRÜMEK YÜRÜMEK YÜRÜMEK", None, False, False, {"YÜRÜMEK": 3}),  # Triple repetition
        ("ARABA^SÜRMEK BILMEK", None, False, False, {}),  # Compound word
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        input_text = test[0]
        expected_tense = test[1]
        expected_question = test[2]
        expected_negative = test[3]
        expected_repetitions = test[4]
        
        result = preprocessor.preprocess(input_text)
        
        # Check assertions
        tense_ok = result.detected_tense == expected_tense
        question_ok = result.is_question == expected_question
        negative_ok = result.is_negative == expected_negative
        repetition_ok = result.repetitions == expected_repetitions
        
        all_ok = tense_ok and question_ok and negative_ok and repetition_ok
        
        status = "PASS" if all_ok else "FAIL"
        if all_ok:
            passed += 1
        else:
            failed += 1
        
        print(f"\n[{status}] Input: {input_text}")
        print(f"  Processed: {result.processed}")
        
        if not tense_ok:
            print(f"  TENSE: got {result.detected_tense}, expected {expected_tense}")
        if not question_ok:
            print(f"  QUESTION: got {result.is_question}, expected {expected_question}")
        if not negative_ok:
            print(f"  NEGATIVE: got {result.is_negative}, expected {expected_negative}")
        if not repetition_ok:
            print(f"  REPETITION: got {result.repetitions}, expected {expected_repetitions}")
        
        if all_ok:
            print(f"  Tense: {result.detected_tense}, Question: {result.is_question}, Negative: {result.is_negative}")
    
    print(f"\nPreprocessor Results: {passed}/{passed+failed} passed")
    return passed, failed


def test_few_shot_builder():
    """Test Few-Shot Builder example selection."""
    print("\n" + "=" * 60)
    print("TEST: FewShotBuilder")
    print("=" * 60)
    
    builder = FewShotBuilder()
    
    test_cases = [
        # (description, params)
        ("Past tense", {"detected_tense": "past"}),
        ("Future tense", {"detected_tense": "future"}),
        ("Question", {"is_question": True}),
        ("Negation", {"is_negative": True}),
        ("Repetition", {"repetitions": {"GEZMEK": 2}}),
        ("Combined: past + question", {"detected_tense": "past", "is_question": True}),
        ("With Hafiza results", {
            "detected_tense": "past",
            "hafiza_results": [
                {"transkripsiyon": "DÜN OKUL GITMEK", "ceviri": "Dün okula gittim.", "similarity": 0.8},
                {"transkripsiyon": "BEN YEMEK YEMEK", "ceviri": "Yemek yedim.", "similarity": 0.7},
            ]
        }),
    ]
    
    passed = 0
    
    for description, params in test_cases:
        print(f"\n[TEST] {description}")
        examples = builder.build_examples(**params)
        
        # Check that examples were generated
        has_examples = "Transkripsiyon:" in examples
        
        if has_examples:
            print("  PASS: Examples generated")
            passed += 1
            # Show preview
            preview = examples[:200] + "..." if len(examples) > 200 else examples
            print(f"  Preview: {preview}")
        else:
            print("  FAIL: No examples generated")
    
    print(f"\nFewShotBuilder Results: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases) - passed


def test_response_parser():
    """Test Response Parser for multi-alternative parsing."""
    print("\n" + "=" * 60)
    print("TEST: ResponseParser")
    print("=" * 60)
    
    parser = ResponseParser()
    
    # Test case 1: Standard format
    response1 = """
## ALTERNATIF 1
Ceviri: Dün arkadaşımla buluştum ve kahve içtik.
Guven: 9/10
Aciklama: DÜN zaman zarfı geçmiş zaman gerektiriyor.

## ALTERNATIF 2
Ceviri: Dün arkadaşla buluşup kahve içtim.
Guven: 8/10
Aciklama: Daha kısa ve günlük dil.

## ALTERNATIF 3
Ceviri: Arkadaşımla dün kahve içmeye gittik.
Guven: 7/10
Aciklama: Farklı cümle yapısı.
"""
    
    # Test case 2: Turkish characters
    response2 = """
## ALTERNATİF 1
Çeviri: Yarın okula gideceğim.
Güven: 9/10
Açıklama: YARIN gelecek zaman gerektiriyor.

## ALTERNATİF 2
Çeviri: Yarın okula gidiyorum.
Güven: 7/10
Açıklama: Planlanan eylem için şimdiki zaman.
"""
    
    # Test case 3: Simple response (fallback)
    response3 = """
Ceviri: Okula gittim.
Guven: 8/10
Aciklama: Basit ceviri.
"""
    
    # Test case 4: Malformed response
    response4 = """
Dün arkadaşımla buluştum.
Bu cümle TID'den çevrildi.
"""
    
    test_cases = [
        ("Standard 3-alternative format", response1, 3),
        ("Turkish characters (ALTERNATİF)", response2, 2),
        ("Simple response (fallback)", response3, 1),
        ("Malformed response", response4, 1),  # Should still extract something
    ]
    
    passed = 0
    
    for description, response, expected_count in test_cases:
        print(f"\n[TEST] {description}")
        result = parser.parse(response)
        
        actual_count = len(result.alternatives)
        
        if actual_count >= expected_count or (expected_count == 1 and actual_count >= 1):
            print(f"  PASS: Found {actual_count} alternatives (expected {expected_count})")
            passed += 1
            
            # Show best translation
            if result.best:
                print(f"  Best: [{result.best.confidence}/10] {result.best.translation}")
        else:
            print(f"  FAIL: Found {actual_count} alternatives (expected {expected_count})")
            print(f"  Parse errors: {result.parse_errors}")
    
    print(f"\nResponseParser Results: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases) - passed


def test_linguistic_context():
    """Test linguistic context building."""
    print("\n" + "=" * 60)
    print("TEST: Linguistic Context Builder")
    print("=" * 60)
    
    test_cases = [
        ("No features", {}),
        ("Past tense (explicit)", {"detected_tense": "past", "tense_source": "explicit"}),
        ("Past tense (inferred)", {"detected_tense": "past", "tense_source": "inferred"}),
        ("Question", {"is_question": True}),
        ("Negation", {"is_negative": True}),
        ("Repetition", {"repetitions": {"GEZMEK": 2}}),
        ("Complex", {
            "detected_tense": "past",
            "is_question": True,
            "repetitions": {"BEKLEMEK": 2},
            "linguistic_hints": {"likely_topic": "ARABA", "verbs": ["ALMAK"]}
        }),
    ]
    
    passed = 0
    
    for description, params in test_cases:
        print(f"\n[TEST] {description}")
        context = build_linguistic_context(**params)
        
        # Basic check: should have content
        has_content = len(context) > 20
        
        if has_content:
            print("  PASS: Context generated")
            passed += 1
            # Show preview
            lines = context.split('\n')[:5]
            for line in lines:
                print(f"    {line}")
        else:
            print("  FAIL: Empty or too short context")
    
    print(f"\nLinguistic Context Results: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases) - passed


def test_markers_replacement():
    """Test special marker replacement in preprocessor."""
    print("\n" + "=" * 60)
    print("TEST: Special Marker Replacement")
    print("=" * 60)
    
    preprocessor = TIDPreprocessor()
    
    test_cases = [
        ("YEMEK YEMEK BITMEK", "_GECMIS_ZAMAN_"),  # BITMEK -> _GECMIS_ZAMAN_
        ("BEN GITMEK DEGIL", "_NEGASYON_"),  # DEGIL -> _NEGASYON_
        ("PARA YOK", "_NEGASYON_"),  # YOK -> _NEGASYON_
        ("GEZMEK GEZMEK", "_TEKRAR"),  # Repetition marker
    ]
    
    passed = 0
    
    for input_text, expected_marker in test_cases:
        print(f"\n[TEST] {input_text} -> should contain {expected_marker}")
        result = preprocessor.preprocess(input_text)
        
        if expected_marker in result.processed:
            print(f"  PASS: Found {expected_marker}")
            print(f"  Processed: {result.processed}")
            passed += 1
        else:
            print(f"  FAIL: {expected_marker} not found in: {result.processed}")
    
    print(f"\nMarker Replacement Results: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases) - passed


def main():
    """Run all linguistic tests."""
    print("=" * 60)
    print("TID LINGUISTIC FEATURE TESTS")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all tests
    p, f = test_preprocessor()
    total_passed += p
    total_failed += f
    
    p, f = test_few_shot_builder()
    total_passed += p
    total_failed += f
    
    p, f = test_response_parser()
    total_passed += p
    total_failed += f
    
    p, f = test_linguistic_context()
    total_passed += p
    total_failed += f
    
    p, f = test_markers_replacement()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    if total_failed == 0:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{total_failed} tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
