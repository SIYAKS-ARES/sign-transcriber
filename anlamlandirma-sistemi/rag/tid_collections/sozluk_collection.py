"""
TID_Sozluk Collection Management
================================
Manages the static dictionary collection in ChromaDB.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.utils import embedding_functions

from rag.config import (
    VECTORSTORE_PATH,
    SOZLUK_COLLECTION,
    EMBEDDING_MODEL,
    DISTANCE_METRIC,
    SOZLUK_TOP_K,
    SIMILARITY_THRESHOLD,
    TID_SOZLUK_PATH,
    MAX_DEFINITIONS_PER_WORD,
)
from rag.preprocessing.cleaning import clean_sozluk_entry, prepare_for_embedding


class SozlukCollection:
    """Manages the TID_Sozluk (dictionary) collection in ChromaDB."""
    
    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize the Sozluk collection.
        
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
            name=SOZLUK_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "TID Kelime Bazli Sozluk Verisi",
                "distance_metric": DISTANCE_METRIC,
                "hnsw:space": DISTANCE_METRIC,
            }
        )
    
    def get_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
    
    def add_entry(self, cleaned_entry: Dict) -> int:
        """
        Add a cleaned dictionary entry to the collection.
        
        Args:
            cleaned_entry: Output from clean_sozluk_entry()
            
        Returns:
            Number of documents added
        """
        documents = prepare_for_embedding(cleaned_entry)
        
        if not documents:
            return 0
        
        # Limit to MAX_DEFINITIONS_PER_WORD
        documents = documents[:MAX_DEFINITIONS_PER_WORD]
        
        self.collection.add(
            documents=[d["document"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
            ids=[d["id"] for d in documents],
        )
        
        return len(documents)
    
    def add_from_json(self, json_path: Path) -> int:
        """
        Add entry from a data.json file.
        
        Args:
            json_path: Path to the data.json file
            
        Returns:
            Number of documents added
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cleaned = clean_sozluk_entry(data)
            return self.add_entry(cleaned)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            return 0
    
    def query_exact(
        self,
        word: str,
        n_results: int = SOZLUK_TOP_K,
    ) -> List[Dict]:
        """
        Query the collection for an exact word match on the 'kelime' metadata field.
        
        Args:
            word: The word to search for (exact match)
            n_results: Maximum number of results to return
            
        Returns:
            List of matching documents (exact matches only)
        """
        import re
        
        word_upper = word.upper().strip()
        word_title = word.strip().title()
        word_lower = word.lower().strip()
        
        # Try different case variations for exact match
        # ChromaDB where filter uses exact string matching
        for search_term in [word_upper, word_title, word_lower, word.strip()]:
            results = self.collection.get(
                where={"kelime": search_term},
                include=["documents", "metadatas"],
                limit=n_results,
            )
            
            if results["ids"]:
                formatted = []
                for i, doc in enumerate(results["documents"]):
                    formatted.append({
                        "document": doc,
                        "similarity": 1.0,  # Exact match = perfect similarity
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        "match_type": "exact",
                    })
                return formatted
        
        # Also try partial match for compound entries like "Abi, Agabey"
        # But only match if the word appears as a COMPLETE word (not inside another word)
        # e.g., "Abi" should match "Abi, Agabey" but NOT "Tabi"
        all_data = self.collection.get(include=["documents", "metadatas"])
        formatted = []
        
        # Pattern: word must be at start/end of string or surrounded by non-letter chars
        # This prevents "abi" matching "Tabi" but allows "Abi" matching "Abi, Agabey"
        word_pattern = re.compile(
            r'(^|[^a-zA-ZğüşöçıİĞÜŞÖÇ])' + re.escape(word_title) + r'($|[^a-zA-ZğüşöçıİĞÜŞÖÇ])',
            re.IGNORECASE
        )
        
        for i, meta in enumerate(all_data["metadatas"] or []):
            kelime = meta.get("kelime", "")
            # Check if the word appears as a complete word in kelime
            if word_pattern.search(kelime):
                formatted.append({
                    "document": all_data["documents"][i],
                    "similarity": 0.95,  # Partial match (compound entry)
                    "metadata": meta,
                    "match_type": "partial",
                })
                if len(formatted) >= n_results:
                    break
        
        return formatted
    
    def query_semantic(
        self, 
        word: str, 
        n_results: int = SOZLUK_TOP_K,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query the collection using semantic (embedding) similarity.
        
        Args:
            word: The word to search for
            n_results: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents with similarity scores
        """
        include = ["documents", "distances"]
        if include_metadata:
            include.append("metadatas")
        
        results = self.collection.query(
            query_texts=[word.upper()],
            n_results=n_results,
            include=include,
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity (for cosine, similarity = 1 - distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                # Skip if below threshold
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                
                entry = {
                    "document": doc,
                    "similarity": similarity,
                    "match_type": "semantic",
                }
                
                if include_metadata and results["metadatas"]:
                    entry["metadata"] = results["metadatas"][0][i]
                
                formatted.append(entry)
        
        return formatted
    
    def query(
        self, 
        word: str, 
        n_results: int = SOZLUK_TOP_K,
        include_metadata: bool = True,
        exact_only: bool = True,
    ) -> List[Dict]:
        """
        Query the dictionary for a word.
        
        Args:
            word: The word to search for
            n_results: Number of results to return
            include_metadata: Whether to include metadata in results
            exact_only: If True, only return exact matches (recommended for dictionary)
                       If False, fall back to semantic search when no exact match
            
        Returns:
            List of matching documents with scores.
            Empty list if no exact match found (when exact_only=True).
        """
        # Try exact match first
        exact_results = self.query_exact(word, n_results=n_results)
        if exact_results:
            return exact_results
        
        # If exact_only mode, don't fall back to semantic search
        # This prevents returning wrong words (e.g., "ABLA" for "ABI")
        if exact_only:
            return []  # Word not found in dictionary
        
        # Fall back to semantic search (only if explicitly requested)
        return self.query_semantic(word, n_results=n_results, include_metadata=include_metadata)
    
    def query_multiple(
        self, 
        words: List[str], 
        n_results_per_word: int = SOZLUK_TOP_K
    ) -> Dict[str, List[Dict]]:
        """
        Query multiple words at once.
        
        Args:
            words: List of words to search for
            n_results_per_word: Number of results per word
            
        Returns:
            Dictionary mapping each word to its results
        """
        results = {}
        for word in words:
            results[word] = self.query(word, n_results=n_results_per_word)
        return results
    
    def delete_all(self) -> None:
        """Delete all documents from the collection (use with caution)."""
        # Get all IDs
        all_data = self.collection.get()
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
    
    def get_all_words(self) -> List[str]:
        """Get all unique words in the collection."""
        all_data = self.collection.get(include=["metadatas"])
        words = set()
        if all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                if metadata and "kelime" in metadata:
                    words.add(metadata["kelime"])
        return sorted(list(words))


def load_all_sozluk_data(sozluk_path: Optional[Path] = None) -> int:
    """
    Load all data.json files from TID_Sozluk_Verileri into the collection.
    
    Args:
        sozluk_path: Path to TID_Sozluk_Verileri folder
        
    Returns:
        Total number of documents added
    """
    from tqdm import tqdm
    
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
    collection = SozlukCollection()
    
    # Check if already populated
    current_count = collection.get_count()
    if current_count > 0:
        print(f"Collection already has {current_count} documents.")
        response = input("Do you want to delete and reload? (y/N): ")
        if response.lower() == 'y':
            collection.delete_all()
            print("Collection cleared.")
        else:
            print("Skipping reload.")
            return current_count
    
    # Load all files
    total_added = 0
    for json_path in tqdm(json_files, desc="Loading sozluk"):
        added = collection.add_from_json(json_path)
        total_added += added
    
    print(f"Total documents added: {total_added}")
    return total_added


if __name__ == "__main__":
    # When run directly, load all data
    load_all_sozluk_data()
