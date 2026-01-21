#!/usr/bin/env python3
"""
Vector Store Baslatma Script'i
Soru CSV'sini yukleyip FAISS index olusturur
"""

import sys
from pathlib import Path

# Proje root'una path ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.services.embedding_service import EmbeddingPipeline


def main():
    """Vectorstore baslat"""
    print("=" * 60)
    print("LGS RAG - Vectorstore Baslatma")
    print("=" * 60)

    # CSV yolu
    csv_path = settings.questions_csv_path
    
    # Alternatif yollar dene
    if not Path(csv_path).exists():
        alt_paths = [
            project_root.parent / "lgs-model" / "data" / "processed" / "dataset_ocr_li.csv",
            project_root / "data" / "dataset_ocr_li.csv",
        ]
        for alt in alt_paths:
            if alt.exists():
                csv_path = alt
                break
        else:
            print(f"HATA: CSV dosyasi bulunamadi!")
            print(f"Beklenen: {settings.questions_csv_path}")
            print(f"Alternatifler: {alt_paths}")
            return 1

    print(f"CSV dosyasi: {csv_path}")

    # Pipeline olustur
    print("\nEmbedding modeli yukleniyor...")
    pipeline = EmbeddingPipeline()
    pipeline.load_model()

    # Sorulari yukle
    print("\nSorular yukleniyor...")
    texts, metadata = EmbeddingPipeline.load_questions_from_csv(str(csv_path))
    print(f"Yuklendi: {len(texts)} soru")

    # Embedding olustur
    print("\nEmbedding'ler olusturuluyor...")
    embeddings = pipeline.embed_batch(texts)
    print(f"Embedding boyutu: {embeddings.shape}")

    # Index olustur
    print("\nFAISS index olusturuluyor...")
    pipeline.build_index(embeddings, metadata)

    # Kaydet
    print("\nIndex kaydediliyor...")
    pipeline.save_index()

    # Test
    print("\n" + "=" * 60)
    print("TEST: Ornek arama")
    print("=" * 60)
    
    results = pipeline.search("EBOB ve EKOK ile ilgili bir problem", top_k=3)
    for i, (score, meta) in enumerate(results, 1):
        print(f"\n[{i}] Skor: {score:.4f}")
        print(f"    Alt Konu: {meta.get('Alt_Konu')}")
        print(f"    Zorluk: {meta.get('Zorluk')}")
        print(f"    Soru: {meta.get('Soru_MetniOCR', '')[:80]}...")

    print("\n" + "=" * 60)
    print("Vectorstore basariyla olusturuldu!")
    print(f"Konum: {settings.vector_store_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

