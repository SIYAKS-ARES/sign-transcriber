#!/usr/bin/env python3
"""
Initialize TID_Hafiza Collection
================================
Script to seed the TID_Hafiza ChromaDB collection with example translations
from TID_Sozluk_Verileri.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tid_collections.hafiza_collection import seed_from_sozluk, HafizaCollection


def main():
    """Main function to initialize the hafiza collection."""
    print("=" * 60)
    print("TID Hafiza Collection Initialization")
    print("=" * 60)
    
    # Check current count
    collection = HafizaCollection()
    current_count = collection.get_count()
    
    if current_count > 0:
        print(f"Collection already has {current_count} documents.")
        response = input("Do you want to clear and reseed? (y/N): ")
        if response.lower() == 'y':
            collection.delete_all()
            print("Collection cleared.")
        else:
            print("Skipping reseed.")
            return
    
    # Seed from sozluk
    total = seed_from_sozluk()
    
    # Verify
    count = collection.get_count()
    
    print("\n" + "=" * 60)
    print(f"Initialization complete!")
    print(f"Total translations in Hafiza: {count}")
    print("=" * 60)
    
    # Test query
    print("\nTest query for 'OKUL GITMEK':")
    results = collection.query("OKUL GITMEK")
    for r in results:
        print(f"  - {r['transkripsiyon'][:50]}...")
        print(f"    Ceviri: {r['ceviri'][:50]}...")
        print(f"    Similarity: {r['similarity']:.3f}")


if __name__ == "__main__":
    main()
