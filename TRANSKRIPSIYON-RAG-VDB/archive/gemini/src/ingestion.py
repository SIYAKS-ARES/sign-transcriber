import os
import json
import uuid
import glob
from tqdm import tqdm
from . import config
from .database import TIDVectorDB

def parse_and_ingest():
    """
    Scans the source directory for data.json files and ingests them into the dictionary collection.
    """
    db = TIDVectorDB()
    source_dir = config.SOURCE_DATA_DIR
    
    print(f"Scanning for data.json files in {source_dir}...")
    
    # Using glob to find all data.json files recursively
    # The structure is SOURCE_DATA_DIR / Word / data.json
    json_files = glob.glob(os.path.join(source_dir, "**", "data.json"), recursive=True)
    
    print(f"Found {len(json_files)} data.json files. Starting ingestion...")
    
    success_count = 0
    error_count = 0

    for json_path in tqdm(json_files, desc="Ingesting Words"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Key Information
            kelime = data.get("kelime")
            if not kelime:
                continue

            # Prepare Metadata (Summary of Meanings)
            anlamlar = data.get("anlamlar", [])
            anlam_summary_parts = []
            
            for anlam in anlamlar:
                tur = anlam.get("tur", "Bilinmiyor")
                aciklama = anlam.get("aciklama", "").strip()
                # Clean up web scraping artifacts if common patterns exist
                # (Simple cleanup for now)
                aciklama = aciklama.replace("\n", " ")
                
                ornek_ceviri = anlam.get("ornek", {}).get("ceviri", "")
                
                part = f"[{tur}] {aciklama}"
                if ornek_ceviri:
                    part += f" (Örnek: {ornek_ceviri})"
                anlam_summary_parts.append(part)
            
            anlam_ozeti = " | ".join(anlam_summary_parts)
            
            # Create a unique ID based on the word to avoid duplicates if re-run, 
            # or just use random UUID. Using word as prefix helps debugging.
            doc_id = f"{kelime}_{uuid.uuid4().hex[:8]}"

            # Add to ChromaDB
            db.add_word(
                word=kelime,
                metadata={
                    "json_dump": json.dumps(data, ensure_ascii=False),
                    "ozet_anlam": anlam_ozeti,
                    "filename": os.path.basename(os.path.dirname(json_path))
                },
                doc_id=doc_id
            )
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            error_count += 1

    print(f"Ingestion Complete. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    parse_and_ingest()
