#!/usr/bin/env python3
"""
Initialize RAG Vectorstore
==========================
Script to initialize or verify the ChromaDB vectorstore for the RAG system.

Usage:
    python scripts/init_vectorstore.py --check    # Check if vectorstore exists
    python scripts/init_vectorstore.py --init     # Initialize from scratch (if needed)
    python scripts/init_vectorstore.py --stats    # Show statistics
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_vectorstore():
    """Check if vectorstore exists and has data."""
    try:
        from rag.config import VECTORSTORE_PATH
        from rag.tid_collections.sozluk_collection import SozlukCollection
        from rag.tid_collections.hafiza_collection import HafizaCollection
        
        print(f"Vectorstore path: {VECTORSTORE_PATH}")
        print(f"Path exists: {VECTORSTORE_PATH.exists()}")
        
        if not VECTORSTORE_PATH.exists():
            print("\nVectorstore bulunamadi!")
            print("Cozum: TRANSKRIPSIYON-RAG-VDB/vectorstore klasorunu kopyalayin:")
            print("  cp -r ../TRANSKRIPSIYON-RAG-VDB/vectorstore .")
            return False
        
        # Check collections
        sozluk = SozlukCollection()
        hafiza = HafizaCollection()
        
        sozluk_count = sozluk.get_count()
        hafiza_count = hafiza.get_count()
        
        print(f"\nSozluk koleksiyonu: {sozluk_count} kayit")
        print(f"Hafiza koleksiyonu: {hafiza_count} kayit")
        
        if sozluk_count == 0:
            print("\nUYARI: Sozluk koleksiyonu bos!")
            return False
        
        print("\nVectorstore hazir!")
        return True
        
    except Exception as e:
        print(f"Hata: {e}")
        return False


def show_stats():
    """Show detailed vectorstore statistics."""
    try:
        from rag.tid_collections.sozluk_collection import SozlukCollection
        from rag.tid_collections.hafiza_collection import HafizaCollection
        
        sozluk = SozlukCollection()
        hafiza = HafizaCollection()
        
        print("=" * 60)
        print("RAG Vectorstore Istatistikleri")
        print("=" * 60)
        
        print(f"\nSozluk (TID_Sozluk):")
        print(f"  - Toplam kayit: {sozluk.get_count()}")
        
        # Test query
        test_results = sozluk.query("OKUL")
        print(f"  - Test sorgusu 'OKUL': {len(test_results)} sonuc")
        
        print(f"\nHafiza (TID_Hafiza):")
        print(f"  - Toplam kayit: {hafiza.get_count()}")
        
        # Test query
        hafiza_results = hafiza.query("BEN OKUL GITMEK", n_results=3)
        print(f"  - Test sorgusu 'BEN OKUL GITMEK': {len(hafiza_results)} sonuc")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Hata: {e}")


def init_vectorstore():
    """Initialize vectorstore from TID_Sozluk_Verileri (if available)."""
    try:
        from rag.config import TID_SOZLUK_PATH, VECTORSTORE_PATH
        
        print(f"TID_Sozluk_Verileri path: {TID_SOZLUK_PATH}")
        print(f"Vectorstore path: {VECTORSTORE_PATH}")
        
        if not TID_SOZLUK_PATH.exists():
            print(f"\nHATA: TID_Sozluk_Verileri bulunamadi: {TID_SOZLUK_PATH}")
            print("Bu klasor olmadan vectorstore olusturulamaz.")
            print("\nAlternatif: Hazir vectorstore'u kopyalayin:")
            print("  cp -r ../TRANSKRIPSIYON-RAG-VDB/vectorstore .")
            return False
        
        print("\nVectorstore olusturuluyor...")
        print("(Bu islem ~5-10 dakika surebilir)\n")
        
        # Import and initialize collections
        from rag.tid_collections.sozluk_collection import SozlukCollection, load_all_sozluk_data
        from rag.tid_collections.hafiza_collection import HafizaCollection, load_hafiza_from_sozluk
        
        # Initialize Sozluk
        print("1. Sozluk koleksiyonu olusturuluyor...")
        load_all_sozluk_data()
        sozluk = SozlukCollection()
        print(f"   Sozluk: {sozluk.get_count()} kayit yuklendi")
        
        # Initialize Hafiza
        print("2. Hafiza koleksiyonu olusturuluyor...")
        load_hafiza_from_sozluk()
        hafiza = HafizaCollection()
        print(f"   Hafiza: {hafiza.get_count()} kayit yuklendi")
        
        print("\nVectorstore basariyla olusturuldu!")
        return True
        
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="RAG Vectorstore Management")
    parser.add_argument("--check", action="store_true", help="Check vectorstore status")
    parser.add_argument("--init", action="store_true", help="Initialize vectorstore")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    if args.init:
        success = init_vectorstore()
        sys.exit(0 if success else 1)
    elif args.stats:
        show_stats()
    else:
        # Default: check
        success = check_vectorstore()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
