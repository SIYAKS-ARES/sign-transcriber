import os
import threading
from typing import List, Tuple

try:
    # scikit-learn TF-IDF ile hafif bir getirici
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # Opsiyonel bağımlılık
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore


_LOCK = threading.Lock()
_VECTORIZER = None
_MATRIX = None
_DOCS: List[str] = []
_DOC_PATHS: List[str] = []


def _read_corpus(kb_dir: str) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    paths: List[str] = []
    if not os.path.isdir(kb_dir):
        return texts, paths
    for root, _, files in os.walk(kb_dir):
        for name in files:
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    paths.append(path)
            except Exception:
                continue
    return texts, paths


def _ensure_index_built() -> None:
    global _VECTORIZER, _MATRIX, _DOCS, _DOC_PATHS
    if _MATRIX is not None:
        return
    kb_dir = os.environ.get("RAG_KB_DIR", os.path.join(os.path.dirname(__file__), "knowledge_base"))
    with _LOCK:
        if _MATRIX is not None:
            return
        docs, paths = _read_corpus(kb_dir)
        _DOCS = docs
        _DOC_PATHS = paths
        if not docs or TfidfVectorizer is None:
            _VECTORIZER = None
            _MATRIX = None
            return
        _VECTORIZER = TfidfVectorizer(min_df=1, max_df=0.95)
        _MATRIX = _VECTORIZER.fit_transform(docs)


def get_retrieved_context(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """Basit TF-IDF ile en alakalı top_k belgeyi döndürür.

    Returns: List of tuples (snippet, path, score)
    """
    if not query or not query.strip():
        return []
    _ensure_index_built()

    if _MATRIX is None or _VECTORIZER is None or cosine_similarity is None or not _DOCS:
        return []

    try:
        q_vec = _VECTORIZER.transform([query])
        sims = cosine_similarity(q_vec, _MATRIX)[0]
    except Exception:
        return []

    # Skorları sırala
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:max(1, top_k)]
    results: List[Tuple[str, str, float]] = []
    for idx, score in ranked:
        text = _DOCS[idx]
        # kısa bir özet/snippet: ilk 500 karakter
        snippet = text.strip().replace("\n", " ")[:500]
        results.append((snippet, _DOC_PATHS[idx], float(score)))
    return results


