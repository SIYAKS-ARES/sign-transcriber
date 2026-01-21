"""
TID RAG System Configuration
============================
Hyperparameter justification and system configuration for the
Iterative Dictionary-Augmented Generation system.
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
TID_SOZLUK_PATH = BASE_DIR.parent / "TID_Sozluk_Verileri"

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
LLM_PROVIDERS = ["openai", "gemini"]
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "gemini")

# =============================================================================
# EMBEDDING MODEL
# Justification: paraphrase-multilingual-MiniLM-L12-v2 secildi cunku:
# 1. 50+ dil destekliyor (Turkce dahil)
# 2. Sentence-level semantic similarity icin optimize edilmis
# 3. 384 boyutlu vektor - hiz/kalite dengesi
# Alternatifler: multilingual-e5-large (daha iyi ama yavas)
# =============================================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# =============================================================================
# DISTANCE METRIC
# Justification: Cosine similarity secildi cunku:
# 1. Vektor yonelimini olcer (semantik benzerlik icin uygun)
# 2. Vektor buyuklugunden bagimsiz - normalize edilmis karsilastirma
# 3. NLP/embedding alaninda standart metrik
# Alternatif: L2 (Euclidean) - mutlak mesafe icin, bu use-case icin uygun degil
# =============================================================================
DISTANCE_METRIC = "cosine"  # ChromaDB: "cosine", "l2", "ip" destekler

# =============================================================================
# COLLECTION NAMES
# =============================================================================
SOZLUK_COLLECTION = "tid_sozluk"
HAFIZA_COLLECTION = "tid_hafiza"

# =============================================================================
# RETRIEVAL PARAMETERS
# Justification:
# - HAFIZA_TOP_K=3: Few-shot learning icin 2-3 ornek optimal (cok fazla noise yaratir)
# - SOZLUK_TOP_K=1: Her kelime icin en yakin eslesen yeterli
# - SIMILARITY_THRESHOLD=0.5: Dusuk benzerlikli sonuclari filtrele
# =============================================================================
HAFIZA_TOP_K = 3
SOZLUK_TOP_K = 1
SIMILARITY_THRESHOLD = 0.5

# =============================================================================
# CONTEXT WINDOW LIMITS
# Justification: LLM token limitleri ve prompt kalitesi dengesi
# - GPT-4o: 128K context, ama uzun promptlar dikkat dagilmasina neden olur
# - Gemini: 1M context, ayni sorun
# - MAX_CONTEXT_TOKENS: 2000 token = ~500 kelime, yeterli baglamsal bilgi
# =============================================================================
MAX_CONTEXT_TOKENS = 2000
MAX_DEFINITIONS_PER_WORD = 2  # Bir kelimenin en fazla 2 anlami prompt'a eklenir

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================
BENCHMARK_TEST_SET = BASE_DIR / "evaluation" / "test_sets" / "test_glosses.json"
ENABLE_BASELINE_COMPARISON = True  # RAG vs Zero-shot karsilastirmasi

# =============================================================================
# DATA CLEANING CONFIGURATION
# =============================================================================
# Boilerplate patterns to remove from scraped data
BOILERPLATE_PATTERNS = [
    r"Güncel Türk İşaret Dili Sözlüğü\n",
    r"Sözlük Kullanımı\n",
    r"Hakkında\n",
    r"Proje Ekibi\n",
    r"İletişim\n",
    r"EN\n",
]

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
