#!/usr/bin/env python3
"""
Test Set Isolation Verifier
============================
Verifies that test set examples are NOT in the RAG Hafiza (memory) collection.
This ensures experiment results are valid and not biased by memorized examples.

Usage:
    python experiments/verify_test_isolation.py
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


def load_test_sets():
    """Load all test sets."""
    test_sets_dir = Path(__file__).parent / "test_sets"
    all_glosses = []
    
    for word_count in [3, 4, 5]:
        test_file = test_sets_dir / f"{word_count}_word_glosses.json"
        if test_file.exists():
            with open(test_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    all_glosses.append({
                        "word_count": word_count,
                        "gloss": item["gloss"],
                        "reference": item["reference"]
                    })
    
    return all_glosses


def check_hafiza_overlap(test_glosses):
    """Check if any test glosses are in the Hafiza collection."""
    try:
        from rag.tid_collections.hafiza_collection import HafizaCollection
        
        hafiza = HafizaCollection()
        hafiza_count = hafiza.get_count()
        
        print(f"Hafiza koleksiyonu: {hafiza_count} kayit")
        print("-" * 60)
        
        overlaps = []
        
        for item in test_glosses:
            gloss = item["gloss"]
            
            # Query Hafiza for similar translations
            results = hafiza.query(gloss, n_results=1)
            
            if results:
                best_match = results[0]
                similarity = 1 - best_match.get("distance", 1)  # Convert distance to similarity
                
                # If similarity is very high (>0.95), it's likely an exact or near match
                if similarity > 0.95:
                    overlaps.append({
                        "test_gloss": gloss,
                        "test_reference": item["reference"],
                        "hafiza_match": best_match.get("transkripsiyon", ""),
                        "hafiza_translation": best_match.get("ceviri", ""),
                        "similarity": similarity,
                    })
        
        return overlaps
        
    except Exception as e:
        print(f"Hafiza kontrolu basarisiz: {e}")
        return None


def main():
    print("=" * 60)
    print("Test Seti Izolasyonu Dogrulama")
    print("=" * 60)
    
    # Load test sets
    test_glosses = load_test_sets()
    print(f"\nToplam test ornegi: {len(test_glosses)}")
    print(f"  - 3 kelime: {sum(1 for g in test_glosses if g['word_count'] == 3)}")
    print(f"  - 4 kelime: {sum(1 for g in test_glosses if g['word_count'] == 4)}")
    print(f"  - 5 kelime: {sum(1 for g in test_glosses if g['word_count'] == 5)}")
    
    print("\n" + "=" * 60)
    print("RAG Hafiza Overlap Kontrolu")
    print("=" * 60 + "\n")
    
    overlaps = check_hafiza_overlap(test_glosses)
    
    if overlaps is None:
        print("RAG sistemi kulanilamiyor, kontrol yapilamadi.")
        return 1
    
    if overlaps:
        print(f"UYARI: {len(overlaps)} test ornegi Hafiza'da bulundu!")
        print("\nOverlap detaylari:")
        for o in overlaps:
            print(f"\n  Test: {o['test_gloss']}")
            print(f"  Hafiza: {o['hafiza_match']}")
            print(f"  Benzerlik: {o['similarity']:.2%}")
        
        print("\n" + "=" * 60)
        print("SONUC: Test seti izole DEGIL!")
        print("Bu ornekleri Hafiza'dan kaldirmayi dusunun.")
        print("=" * 60)
        return 1
    else:
        print("Hicbir test ornegi Hafiza'da bulunamadi.")
        print("\n" + "=" * 60)
        print("SONUC: Test seti izole - deneyler gecerli!")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
