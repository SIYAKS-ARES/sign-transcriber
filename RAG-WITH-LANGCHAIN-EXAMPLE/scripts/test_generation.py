#!/usr/bin/env python3
"""
Soru Uretim Test Script'i
Manuel olarak soru uretimi test eder
"""

import sys
import asyncio
import json
from pathlib import Path

# Proje root'una path ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.services.embedding_service import EmbeddingPipeline, set_embedding_pipeline
from src.services.filter_service import FilterService
from src.services.retriever import QuestionRetriever
from src.services.output_formatter import RAGOutputFormatter, format_question_output
from src.services.prompt_builder import PromptBuilder
from src.services.llm_client import LLMClientFactory
from src.services.question_generator import QuestionGenerator


async def test_generation():
    """Soru uretim testi"""
    print("=" * 60)
    print("LGS RAG - Soru Uretim Testi")
    print("=" * 60)

    # Servisleri baslat
    print("\n1. Servisler baslatiliyor...")

    # Embedding pipeline
    pipeline = EmbeddingPipeline()
    pipeline.load_model()

    # Index yukle veya olustur
    index_path = Path(settings.vector_store_path) / "faiss.index"
    if index_path.exists():
        pipeline.load_index()
    else:
        print("   Index bulunamadi, once init_vectorstore.py calistirin!")
        return 1

    set_embedding_pipeline(pipeline)

    # Filter service
    filter_service = FilterService(list(pipeline.id_to_metadata.values()))
    print(f"   Filter service: {len(filter_service.all_metadata)} soru")

    # Retriever
    retriever = QuestionRetriever(pipeline, filter_service)
    print("   Retriever hazir")

    # LLM Client
    print("\n2. LLM Client baslatiliyor...")
    try:
        llm_client = LLMClientFactory.create()
        print(f"   LLM: {settings.llm_provider} / {settings.llm_model}")
    except Exception as e:
        print(f"   HATA: LLM client baslatılamadi: {e}")
        print("   .env dosyasinda OPENAI_API_KEY ayarlayin!")
        return 1

    # Generator
    generator = QuestionGenerator(
        retriever=retriever,
        formatter=RAGOutputFormatter(format_type="markdown"),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
    )
    print("   Generator hazir")

    # Test kombinasyonlari
    test_combinations = [
        {"alt_konu": "ebob_ekok", "zorluk": 3, "gorsel_tipi": "yok", "lgs_skor": 0.85},
        {"alt_konu": "carpanlar", "zorluk": 2, "gorsel_tipi": "yok", "lgs_skor": 0.80},
    ]

    print("\n3. Soru uretim testleri...")
    print("=" * 60)

    for i, combo in enumerate(test_combinations, 1):
        print(f"\n--- Test {i}: {combo['alt_konu']}, Z:{combo['zorluk']} ---")

        try:
            question = await generator.generate_question(
                combination=combo,
                style_instruction="Gunluk hayattan ornek kullan.",
            )

            print(format_question_output(question.model_dump()))

            # JSON olarak da kaydet
            output_file = project_root / "data" / "generated_questions" / f"test_{i}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(question.model_dump(), f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\nKaydedildi: {output_file}")

        except Exception as e:
            print(f"\nHATA: {e}")

    print("\n" + "=" * 60)
    print("Test tamamlandi!")
    print("=" * 60)

    return 0


def main():
    """Ana fonksiyon"""
    return asyncio.run(test_generation())


if __name__ == "__main__":
    sys.exit(main())

