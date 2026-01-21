
import sys
import os
from unittest.mock import MagicMock

# Mock heavy dependencies to avoid import errors in simple logic tests
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["rag.retriever"] = MagicMock()
sys.modules["rag.retriever.dual_retriever"] = MagicMock()

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.getcwd(), 'anlamlandirma-sistemi'))

from rag.preprocessing.tid_preprocessor import TIDPreprocessor
from rag.prompt_builder.system_instructions import build_dynamic_system_instruction

def test_grammar_detection():
    preprocessor = TIDPreprocessor()
    
    test_cases = [
        {
            "input": "ARKADAS BERABER SINEMA GITMEK",
            "expected_grammar": {"BERABER": "vasita_birliktelik"},
            "expected_verb_classes": ["yonelimli"], # GITMEK
        },
        {
            "input": "GITMEK LAZIM",
            "expected_grammar": {"LAZIM": "gereklilik"},
            "expected_verb_classes": ["yonelimli"], # GITMEK
        },
        {
            "input": "BIKMAK",
            "expected_grammar": {},
            "expected_verb_classes": ["duygu"], # BIKMAK
        },
        {
            "input": "GORMEK ICIN",
            "expected_grammar": {"ICIN": "amac_sonuc"},
            "expected_verb_classes": [], 
        }
    ]
    
    print("=== TEST BASLIYOR ===")
    
    for case in test_cases:
        text = case["input"]
        print(f"\nInput: {text}")
        
        result = preprocessor.preprocess(text)
        
        # Check grammar hints
        print(f"Detected Grammar: {result.grammar_hints}")
        for k, v in case["expected_grammar"].items():
            if k not in result.grammar_hints or result.grammar_hints[k] != v:
                print(f"FAILED: Expected {k}->{v}")
            else:
                print(f"PASSED: {k}->{v}")

        # Check verb classes
        print(f"Detected Verb Classes: {result.verb_classes}")
        for v_class in case["expected_verb_classes"]:
            if v_class not in result.verb_classes:
                print(f"FAILED: Expected class {v_class}")
            else:
                print(f"PASSED: Class {v_class}")
                
        # Check Prompt Generation
        prompt = build_dynamic_system_instruction(
            grammar_hints=result.grammar_hints,
            verb_classes=result.verb_classes
        )
        
        if result.grammar_hints and "GRAMER IPUCLARI" not in prompt:
             print("FAILED: Prompt missing GRAMER IPUCLARI")
        elif result.grammar_hints:
             print("PASSED: Prompt has GRAMER IPUCLARI")
             
        if result.verb_classes and "FIIL SINIFI IPUCLARI" not in prompt:
             print("FAILED: Prompt missing FIIL SINIFI IPUCLARI")
        elif result.verb_classes:
             print("PASSED: Prompt has FIIL SINIFI IPUCLARI")

    print("\n=== TEST BITTI ===")

if __name__ == "__main__":
    test_grammar_detection()
