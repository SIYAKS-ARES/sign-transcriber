import chromadb
from chromadb.utils import embedding_functions
from . import config
import json

class TIDVectorDB:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TIDVectorDB, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the ChromaDB client and collections."""
        print(f"Initializing ChromaDB at {config.VECTOR_DB_PATH}...")
        self.client = chromadb.PersistentClient(path=str(config.VECTOR_DB_PATH))
        
        # Default embedding function (can be swapped later)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize Collections
        self.sozluk_collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME_SOZLUK,
            embedding_function=self.emb_fn,
            metadata={"description": "TID Glossary Data"}
        )
        
        self.hafiza_collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME_HAFIZA,
            embedding_function=self.emb_fn,
            metadata={"description": "TID Translation Memory"}
        )
        print("Collections ready.")

    def add_word(self, word: str, metadata: dict, doc_id: str):
        """Add a single word to the dictionary collection."""
        self.sozluk_collection.upsert(
            documents=[word],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def add_sentence_memory(self, gloss: str, translation: str, doc_id: str):
        """Add a gloss-translation pair to the memory collection."""
        self.hafiza_collection.upsert(
            documents=[gloss],
            metadatas=[{"dogru_ceviri": translation}],
            ids=[doc_id]
        )

    def query_word(self, word: str, n_results=1):
        """Query the dictionary for a word."""
        return self.sozluk_collection.query(
            query_texts=[word],
            n_results=n_results
        )

    def query_memory(self, gloss_sentence: str, n_results=2):
        """Query the memory for similar gloss sentences."""
        return self.hafiza_collection.query(
            query_texts=[gloss_sentence],
            n_results=n_results
        )
