"""
FastAPI Ana Uygulama
LGS RAG Soru Uretim API'si
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..models.api_models import (
    GenerateRequest,
    GenerateResponse,
    BatchGenerateRequest,
    BatchGenerateResponse,
    HealthResponse,
    ErrorResponse,
)
from ..models.config_schema import ConfigSchema
from ..models.combination import CombinationFilters
from ..services.embedding_service import (
    EmbeddingPipeline,
    set_embedding_pipeline,
)
from ..services.filter_service import FilterService
from ..services.retriever import QuestionRetriever
from ..services.output_formatter import RAGOutputFormatter
from ..services.prompt_builder import PromptBuilder
from ..services.llm_client import LLMClientFactory
from ..services.question_generator import QuestionGenerator, QuestionGenerationError
from ..services.combination_selector import CombinationSelector
from ..services.diversity_service import DiversityService


# Global instances
_generator: QuestionGenerator = None
_selector: CombinationSelector = None
_diversity: DiversityService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama yasam dongusu
    Baslangicta servisleri yukle
    """
    global _generator, _selector, _diversity

    print("Servisler baslatiliyor...")

    try:
        # Embedding pipeline
        csv_path = settings.questions_csv_path
        if not Path(csv_path).exists():
            # Varsayilan yol dene
            # From src/api/main.py: parent -> api/, parent -> src/, parent -> 06-RAG-WITH-LANGCHAIN/, parent -> workspace root
            csv_path = Path(__file__).parent.parent.parent.parent / "lgs-model" / "data" / "processed" / "dataset_ocr_li.csv"

        pipeline = EmbeddingPipeline()
        pipeline.load_model()

        # Index mevcut mu kontrol et
        index_path = Path(settings.vector_store_path) / "faiss.index"
        if index_path.exists():
            pipeline.load_index()
        else:
            print(f"Index bulunamadi, yeni olusturuluyor: {csv_path}")
            texts, metadata = EmbeddingPipeline.load_questions_from_csv(str(csv_path))
            embeddings = pipeline.embed_batch(texts)
            pipeline.build_index(embeddings, metadata)
            pipeline.save_index()

        set_embedding_pipeline(pipeline)

        # Filter service
        filter_service = FilterService(list(pipeline.id_to_metadata.values()))

        # Retriever
        retriever = QuestionRetriever(pipeline, filter_service)

        # LLM Client (cache'i temizle, sonra yeni instance olustur)
        LLMClientFactory.clear_cache()
        llm_client = LLMClientFactory.create()

        # Generator
        _generator = QuestionGenerator(
            retriever=retriever,
            formatter=RAGOutputFormatter(format_type="markdown"),
            prompt_builder=PromptBuilder(),
            llm_client=llm_client,
        )

        # Combination selector (configs.json varsa onu kullan, yoksa mock)
        config_path = Path(settings.configs_path)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                config = ConfigSchema(**config_data)
                print(f"Config yuklendi: {config_path}")
            except Exception as e:
                print(f"Config dosyasi okunamadi, mock config kullaniliyor: {e}")
                config = ConfigSchema.create_mock()
        else:
            print(f"Config dosyasi bulunamadi, mock config kullaniliyor: {config_path}")
            config = ConfigSchema.create_mock()

        _selector = CombinationSelector(config)

        # Diversity service
        _diversity = DiversityService()

        print("Servisler hazir!")

    except Exception as e:
        print(f"Servis baslatma hatasi: {e}")
        # Mock modda calis
        _generator = None
        _selector = None
        _diversity = DiversityService()

    yield

    # Cleanup
    print("Servisler kapatiliyor...")


# FastAPI app
app = FastAPI(
    title="LGS RAG API",
    description="LGS matematik sorulari ureten RAG sistemi",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Ana sayfa"""
    return {
        "message": "LGS RAG API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Sistem saglik kontrolu
    """
    return HealthResponse(
        status="healthy" if _generator else "degraded",
        version="1.0.0",
        timestamp=datetime.now(),
        components={
            "generator": "ready" if _generator else "not_initialized",
            "selector": "ready" if _selector else "not_initialized",
            "diversity": "ready",
        },
    )


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_question(request: GenerateRequest):
    """
    Tek soru uret
    
    - **filters**: Kombinasyon filtreleri (opsiyonel)
    - **specific_combination**: Spesifik kombinasyon (opsiyonel)
    - **style_instruction**: Stil talimati (opsiyonel)
    """
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Soru uretim servisi hazir degil",
        )

    try:
        # Kombinasyon sec
        if request.specific_combination:
            combination = request.specific_combination
        elif _selector:
            filters = None
            if request.filters:
                filters = CombinationFilters(**request.filters.model_dump())
            combo = _selector.select(filters=filters)
            combination = combo.model_dump()
        else:
            # Default kombinasyon
            combination = {
                "alt_konu": "ebob_ekok",
                "zorluk": 3,
                "gorsel_tipi": "yok",
                "lgs_skor": 0.8,
            }

        # Stil
        style = request.style_instruction
        if not style and _diversity:
            style = _diversity.get_random_style()

        # Uret
        question = await _generator.generate_question(
            combination=combination,
            style_instruction=style,
        )

        # Tekrar takibi
        if _diversity:
            _diversity.add_to_history(f"{question.hikaye} {question.soru}")

        return GenerateResponse(
            success=True,
            data=question,
            metadata={
                "combination": combination,
                "style": style,
            },
        )

    except QuestionGenerationError as e:
        return GenerateResponse(
            success=False,
            error=str(e),
            raw_response=getattr(e, 'raw_response', None),
            metadata={"combination": combination if "combination" in locals() else None},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/batch", response_model=BatchGenerateResponse, tags=["Generation"])
async def generate_batch(request: BatchGenerateRequest):
    """
    Toplu soru uretimi
    
    - **count**: Uretilecek soru sayisi (1-50)
    - **filters**: Kombinasyon filtreleri (opsiyonel)
    - **ensure_diversity**: Cesitlilik garantisi
    """
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Soru uretim servisi hazir degil",
        )

    try:
        questions = []
        failed = 0

        # Kombinasyonlari sec
        if _selector:
            filters = None
            if request.filters:
                filters = CombinationFilters(**request.filters.model_dump())
            combinations = _selector.select_multiple(
                count=request.count,
                filters=filters,
                ensure_diversity=request.ensure_diversity,
            )
        else:
            # Default kombinasyonlar
            combinations = [
                {"alt_konu": "ebob_ekok", "zorluk": 3, "gorsel_tipi": "yok"}
                for _ in range(request.count)
            ]

        # Her kombinasyon icin uret
        for combo in combinations:
            try:
                style = None
                if request.ensure_diversity and _diversity:
                    style = _diversity.get_random_style()

                question = await _generator.generate_question(
                    combination=combo.model_dump() if hasattr(combo, "model_dump") else combo,
                    style_instruction=style,
                )
                questions.append(question.model_dump())

            except Exception as e:
                print(f"Uretim hatasi: {e}")
                failed += 1

        return BatchGenerateResponse(
            success=True,
            data={
                "questions": questions,
                "stats": {
                    "requested": request.count,
                    "generated": len(questions),
                    "failed": failed,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/combinations", tags=["Combinations"])
async def list_combinations():
    """
    Mevcut kombinasyonlari listele
    """
    if _selector is None:
        return {
            "message": "Kombinasyon secici hazir degil",
            "combinations": [],
        }

    combinations = [c.model_dump() for c in _selector.combinations]
    stats = _selector.get_statistics()

    return {
        "total": len(combinations),
        "combinations": combinations[:20],  # Ilk 20
        "statistics": stats,
    }


@app.get("/api/v1/stats", tags=["System"])
async def get_stats():
    """
    Sistem istatistikleri
    """
    stats = {
        "diversity": _diversity.get_stats() if _diversity else None,
        "selector": _selector.get_statistics() if _selector else None,
    }

    return stats


# CLI icin
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

