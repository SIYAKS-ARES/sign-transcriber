"""
R-01: Embedding Pipeline Kurulumu
Turkce embedding modeli ve FAISS index yonetimi
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from ..config import settings


class EmbeddingPipeline:
    """
    Embedding ve FAISS index yonetimi
    
    Gorevler:
    1. Turkce embedding modeli yukleme
    2. Metin vektorlestirme
    3. FAISS index olusturma/yukleme
    4. Benzerlik aramasi
    """

    # Turkce embedding modelleri
    TURKISH_MODELS = [
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",  # Birincil
        "dbmdz/bert-base-turkish-cased",  # Alternatif
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Fallback
    ]

    def __init__(self, model_name: Optional[str] = None):
        """
        Embedding pipeline baslatici
        
        Args:
            model_name: Kullanilacak model adi (None ise ayarlardan alinir)
        """
        self.model_name = model_name or self._select_model()
        self.model: Optional[SentenceTransformer] = None
        self.dimension: int = 0
        self.index: Optional[faiss.Index] = None
        self.id_to_metadata: dict[int, dict] = {}
        self._cache: dict[str, np.ndarray] = {}

    def _select_model(self) -> str:
        """
        Uygun embedding modelini sec
        
        OpenAI embedding kullaniliyorsa None doner
        HuggingFace kullaniliyorsa Turkce model secer
        """
        if settings.embedding_provider == "openai":
            return settings.embedding_model
        return self.TURKISH_MODELS[0]

    def load_model(self):
        """
        Embedding modelini yukle
        """
        if settings.embedding_provider == "openai":
            # OpenAI embeddings icin model yukleme gerekmiyor
            # langchain_rag.py'de OpenAIEmbeddings kullaniliyor
            print(f"OpenAI Embedding: {self.model_name}")
            # OpenAI embedding model dimension mapping
            OPENAI_DIMENSIONS = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            # Default to 1536 if model not in mapping
            self.dimension = OPENAI_DIMENSIONS.get(self.model_name, 1536)
        else:
            print(f"Model yukleniyor: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Model yuklendi. Dimension: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """
        Tek metin icin embedding olustur
        
        Args:
            text: Giris metni
            
        Returns:
            Embedding vektoru (1D numpy array)
        """
        # Cache kontrol
        text_hash = str(hash(text))
        if text_hash in self._cache:
            return self._cache[text_hash]

        if self.model is None:
            raise ValueError("Model yuklenmedi. Once load_model() cagirin.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Cosine similarity icin
        )

        # Cache'e ekle
        self._cache[text_hash] = embedding
        
        return embedding

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Toplu embedding olusturma
        
        Args:
            texts: Metin listesi
            show_progress: Ilerleme cubugu goster
            
        Returns:
            Embedding matrisi (N x D)
        """
        if self.model is None:
            raise ValueError("Model yuklenmedi. Once load_model() cagirin.")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        return embeddings

    def build_index(self, embeddings: np.ndarray, metadata_list: list[dict]):
        """
        FAISS index olustur
        
        Args:
            embeddings: Embedding matrisi (N x D)
            metadata_list: Her embedding icin metadata listesi
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embedding ve metadata sayilari eslesmiyor")

        # Inner Product index (normalize edilmis vektorler icin cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Embedding'leri ekle
        embeddings_float32 = embeddings.astype("float32")
        self.index.add(embeddings_float32)

        # Metadata mapping
        self.id_to_metadata = {i: meta for i, meta in enumerate(metadata_list)}

        print(f"Index olusturuldu: {self.index.ntotal} vektor ({self.dimension}D)")

    def save_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Index ve metadata'yi kaydet
        
        Args:
            index_path: FAISS index dosya yolu
            metadata_path: Metadata pickle dosya yolu
        """
        base_path = Path(settings.vector_store_path)
        base_path.mkdir(parents=True, exist_ok=True)

        index_file = Path(index_path) if index_path else base_path / "faiss.index"
        metadata_file = Path(metadata_path) if metadata_path else base_path / "metadata.pkl"

        # FAISS index kaydet
        faiss.write_index(self.index, str(index_file))
        print(f"Index kaydedildi: {index_file}")

        # Metadata kaydet
        with open(metadata_file, "wb") as f:
            pickle.dump(self.id_to_metadata, f)
        print(f"Metadata kaydedildi: {metadata_file}")

    def load_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Index ve metadata'yi yukle
        
        Args:
            index_path: FAISS index dosya yolu
            metadata_path: Metadata pickle dosya yolu
        """
        base_path = Path(settings.vector_store_path)

        index_file = Path(index_path) if index_path else base_path / "faiss.index"
        metadata_file = Path(metadata_path) if metadata_path else base_path / "metadata.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"Index dosyasi bulunamadi: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata dosyasi bulunamadi: {metadata_file}")

        # FAISS index yukle
        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d
        print(f"Index yuklendi: {self.index.ntotal} vektor")

        # Metadata yukle
        with open(metadata_file, "rb") as f:
            self.id_to_metadata = pickle.load(f)
        print(f"Metadata yuklendi: {len(self.id_to_metadata)} kayit")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_indices: Optional[list[int]] = None,
    ) -> list[tuple[float, dict]]:
        """
        Benzer sorulari ara
        
        Args:
            query: Arama sorgusu
            top_k: Dondurelecek sonuc sayisi
            filter_indices: Arama yapilacak index'ler (None = tumu)
            
        Returns:
            [(skor, metadata), ...] listesi
        """
        if self.index is None:
            raise ValueError("Index yuklenmedi. Once load_index() veya build_index() cagirin.")

        # Query embedding
        query_embedding = self.embed(query)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Arama
        if filter_indices is not None and len(filter_indices) > 0:
            # Filtrelenmis arama: Sadece belirli index'lerde ara
            search_k = min(top_k * 3, self.index.ntotal)  # Daha fazla ara, sonra filtrele
            scores, indices = self.index.search(query_embedding, search_k)

            # Filtreleme
            results = []
            filter_set = set(filter_indices)
            for score, idx in zip(scores[0], indices[0]):
                if idx in filter_set and idx in self.id_to_metadata:
                    results.append((float(score), self.id_to_metadata[idx]))
                    if len(results) >= top_k:
                        break
        else:
            # Tum index'te ara
            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx in self.id_to_metadata:
                    results.append((float(score), self.id_to_metadata[idx]))

        return results

    def similarity(self, text1: str, text2: str) -> float:
        """
        Iki metin arasindaki benzerlik skoru
        
        Args:
            text1: Birinci metin
            text2: Ikinci metin
            
        Returns:
            Cosine similarity (0-1)
        """
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)

        # Cosine similarity (normalize edilmis vektorler icin dot product)
        return float(np.dot(vec1, vec2))

    @staticmethod
    def load_questions_from_csv(csv_path: str) -> tuple[list[str], list[dict]]:
        """
        CSV'den soru metinleri ve metadata'lari yukle
        
        Args:
            csv_path: CSV dosya yolu
            
        Returns:
            (texts, metadata_list) tuple
        """
        df = pd.read_csv(csv_path)

        texts = []
        metadata_list = []

        for idx, row in df.iterrows():
            text = str(row.get("Soru_MetniOCR", ""))
            if not text.strip():
                continue

            texts.append(text)
            metadata_list.append({
                "idx": idx,
                "Soru_MetniOCR": text,
                "Alt_Konu": row.get("Alt_Konu", ""),
                "Zorluk": int(row.get("Zorluk", 1)),
                "Gorsel_Tipi": row.get("Gorsel_Tipi", "yok"),
                "Kaynak_Tipi": row.get("Kaynak_Tipi", ""),
                "egitim_agirligi": float(row.get("egitim_agirligi", 0.5)),
                "is_LGS": int(row.get("is_LGS", 0)),
                "ocr_kelime_sayisi": row.get("ocr_kelime_sayisi"),
                "ocr_karakter_sayisi": row.get("ocr_karakter_sayisi"),
                "ocr_rakam_sayisi": row.get("ocr_rakam_sayisi"),
                "ocr_cok_adimli": row.get("ocr_cok_adimli"),
            })

        return texts, metadata_list


def initialize_vectorstore(csv_path: str, force_rebuild: bool = False) -> EmbeddingPipeline:
    """
    Vectorstore baslat (yukle veya olustur)
    
    Args:
        csv_path: Soru CSV dosya yolu
        force_rebuild: Mevcut index'i yeniden olustur
        
    Returns:
        Baslatilmis EmbeddingPipeline
    """
    pipeline = EmbeddingPipeline()
    
    index_path = Path(settings.vector_store_path) / "faiss.index"
    
    if index_path.exists() and not force_rebuild:
        print("Mevcut index yukleniyor...")
        pipeline.load_model()
        pipeline.load_index()
    else:
        print("Yeni index olusturuluyor...")
        pipeline.load_model()
        
        # CSV'den yukle
        texts, metadata_list = EmbeddingPipeline.load_questions_from_csv(csv_path)
        print(f"Yuklendi: {len(texts)} soru")
        
        # Embedding olustur
        embeddings = pipeline.embed_batch(texts)
        
        # Index olustur ve kaydet
        pipeline.build_index(embeddings, metadata_list)
        pipeline.save_index()
    
    return pipeline


# Singleton instance
_pipeline_instance: Optional[EmbeddingPipeline] = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """
    Global embedding pipeline instance dondur
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        raise ValueError("Embedding pipeline baslatilmadi. Once initialize_vectorstore() cagirin.")
    return _pipeline_instance


def set_embedding_pipeline(pipeline: EmbeddingPipeline):
    """
    Global embedding pipeline instance ayarla
    """
    global _pipeline_instance
    _pipeline_instance = pipeline

