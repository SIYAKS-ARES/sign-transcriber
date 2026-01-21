"""
TID_Hafiza Collection Management
================================
Manages the dynamic translation memory collection in ChromaDB.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import chromadb
from chromadb.utils import embedding_functions

from rag.config import (
    VECTORSTORE_PATH,
    HAFIZA_COLLECTION,
    EMBEDDING_MODEL,
    DISTANCE_METRIC,
    HAFIZA_TOP_K,
    SIMILARITY_THRESHOLD,
    TID_SOZLUK_PATH,
)


class HafizaCollection:
    """Manages the TID_Hafiza (translation memory) collection in ChromaDB."""
    
    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize the Hafiza collection.
        
        Args:
            persist_path: Path to ChromaDB storage. Defaults to config value.
        """
        self.persist_path = persist_path or VECTORSTORE_PATH
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        
        # Initialize embedding function (multilingual for Turkish support)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=HAFIZA_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "TID Transkripsiyon ve Ceviri Hafizasi",
                "distance_metric": DISTANCE_METRIC,
                "hnsw:space": DISTANCE_METRIC,
            }
        )
    
    def get_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
    
    def add_translation(
        self,
        transkripsiyon: str,
        ceviri: str,
        provider: str = "manual",
        confidence: float = 1.0,
    ) -> str:
        """
        Add a verified translation to the memory.
        
        Args:
            transkripsiyon: The TID gloss transcription
            ceviri: The verified Turkish translation
            provider: LLM provider used (or "manual" for human input)
            confidence: Confidence score (0-1)
            
        Returns:
            ID of the added document
        """
        doc_id = f"hafiza_{uuid.uuid4().hex[:12]}"
        
        self.collection.add(
            documents=[transkripsiyon.upper()],  # Embed the transcription
            metadatas=[{
                "transkripsiyon": transkripsiyon.upper(),
                "ceviri": ceviri,
                "provider": provider,
                "confidence": confidence,
                "onay_tarihi": datetime.now().isoformat(),
            }],
            ids=[doc_id],
        )
        
        return doc_id
    
    def query(
        self,
        transkripsiyon: str,
        n_results: int = HAFIZA_TOP_K,
    ) -> List[Dict]:
        """
        Query for similar transcriptions.
        
        Args:
            transkripsiyon: The transcription to search for
            n_results: Number of results to return
            
        Returns:
            List of similar translations with scores
        """
        results = self.collection.query(
            query_texts=[transkripsiyon.upper()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                # Skip if below threshold
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                
                entry = {
                    "transkripsiyon": doc,
                    "similarity": similarity,
                    "ceviri": results["metadatas"][0][i]["ceviri"],
                    "provider": results["metadatas"][0][i]["provider"],
                    "onay_tarihi": results["metadatas"][0][i].get("onay_tarihi", ""),
                }
                formatted.append(entry)
        
        return formatted
    
    def delete_by_id(self, doc_id: str) -> bool:
        """Delete a translation by its ID."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def get_all(self) -> List[Dict]:
        """Get all translations in the collection."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        
        results = []
        if all_data["documents"]:
            for i, doc in enumerate(all_data["documents"]):
                results.append({
                    "id": all_data["ids"][i],
                    "transkripsiyon": doc,
                    "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {},
                })
        
        return results
    
    def delete_all(self) -> None:
        """Delete all documents from the collection (use with caution)."""
        all_data = self.collection.get()
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])


def seed_from_sozluk(sozluk_path: Optional[Path] = None) -> int:
    """
    Seed the Hafiza collection with example translations from TID_Sozluk_Verileri.
    
    This uses the example sentences found in the dictionary entries as
    initial translation memory.
    
    Args:
        sozluk_path: Path to TID_Sozluk_Verileri folder
        
    Returns:
        Number of translations added
    """
    from tqdm import tqdm
    from preprocessing.cleaning import clean_sozluk_entry
    
    path = sozluk_path or TID_SOZLUK_PATH
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"TID_Sozluk_Verileri not found at {path}")
    
    # Find all data.json files
    json_files = list(path.glob("*/data.json"))
    
    if not json_files:
        raise ValueError(f"No data.json files found in {path}")
    
    print(f"Found {len(json_files)} data.json files")
    
    # Initialize collection
    collection = HafizaCollection()
    
    # Track seen transcriptions to avoid duplicates
    seen_trans = set()
    added = 0
    
    for json_path in tqdm(json_files, desc="Seeding hafiza"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cleaned = clean_sozluk_entry(data)
            
            for anlam in cleaned["anlamlar"]:
                trans = anlam["transkripsiyon"]
                ceviri = anlam["ceviri"]
                
                # Skip empty or duplicate
                if not trans or not ceviri or trans in seen_trans:
                    continue
                
                # Skip very short transcriptions (likely incomplete)
                if len(trans.split()) < 2:
                    continue
                
                seen_trans.add(trans)
                collection.add_translation(
                    transkripsiyon=trans,
                    ceviri=ceviri,
                    provider="tid_sozluk",
                    confidence=1.0,
                )
                added += 1
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    print(f"\nTotal translations added to Hafiza: {added}")
    return added


if __name__ == "__main__":
    # When run directly, seed from sozluk
    seed_from_sozluk()
