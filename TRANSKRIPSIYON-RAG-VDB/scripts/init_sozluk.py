#!/usr/bin/env python3
"""
Initialize TID_Sozluk Collection
================================
Script to populate the TID_Sozluk ChromaDB collection from TID_Sozluk_Verileri.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tid_collections.sozluk_collection import load_all_sozluk_data, SozlukCollection


def main():
    """Main function to initialize the sozluk collection."""
    print("=" * 60)
    print("TID Sozluk Collection Initialization")
    print("=" * 60)
    
    # Load all data
    total = load_all_sozluk_data()
    
    # Verify
    collection = SozlukCollection()
    count = collection.get_count()
    
    print("\n" + "=" * 60)
    print(f"Initialization complete!")
    print(f"Total documents in collection: {count}")
    print("=" * 60)
    
    # Test query
    print("\nTest query for 'AGAC':")
    results = collection.query("AGAC")
    for r in results:
        print(f"  - {r['metadata']['kelime']}: {r['metadata']['tur']}")
        print(f"    Similarity: {r['similarity']:.3f}")
        print(f"    Aciklama: {r['metadata']['aciklama'][:100]}...")


if __name__ == "__main__":
    main()
