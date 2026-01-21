import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = DATA_DIR / "tid_vector_db"

# Source Data Directory (Relative to the project root, as per user workspace)
# Assuming TRANSKRIPSIYON-RAG-VDB is a sibling of TID_Sozluk_Verileri inside sign-transcriber
PROJECT_ROOT = BASE_DIR.parent
SOURCE_DATA_DIR = PROJECT_ROOT / "TID_Sozluk_Verileri"

# ChromaDB Settings
COLLECTION_NAME_SOZLUK = "tid_sozluk"
COLLECTION_NAME_HAFIZA = "tid_hafiza"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
