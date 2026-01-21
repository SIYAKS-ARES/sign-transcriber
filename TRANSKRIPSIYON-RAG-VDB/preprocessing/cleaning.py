"""
Data Preprocessing Pipeline for TID_Sozluk_Verileri
====================================================
Cleans and normalizes data from web-scraped JSON files.
"""

import re
from typing import Dict, List, Optional


def remove_html_tags(text: str) -> str:
    """Remove any HTML tags from text."""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text)


def remove_boilerplate(text: str) -> str:
    """
    Remove common boilerplate text from scraped data.
    This includes header/footer content that appears in every entry.
    """
    if not text:
        return ""
    
    # Common boilerplate patterns from TID Sozluk website
    boilerplate_patterns = [
        r"Güncel Türk İşaret Dili Sözlüğü\s*",
        r"Sözlük Kullanımı\s*",
        r"Hakkında\s*",
        r"Proje Ekibi\s*",
        r"İletişim\s*",
        r"\bEN\b\s*",  # English language toggle
    ]
    
    result = text
    for pattern in boilerplate_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    return result.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize multiple whitespace characters to single spaces."""
    if not text:
        return ""
    # Replace multiple newlines/spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_word_type(aciklama: str) -> str:
    """
    Extract word type (Ad, Eylem, Sifat, etc.) from the description.
    The format is typically: "1) Eylem description..." or "Ad description..."
    """
    if not aciklama:
        return "Bilinmiyor"
    
    # Common word types in Turkish
    word_types = [
        "Eylem",  # Verb
        "Ad",     # Noun
        "Sıfat",  # Adjective
        "Zarf",   # Adverb
        "Bağlaç", # Conjunction
        "Ünlem",  # Interjection
        "Zamir",  # Pronoun
        "Edat",   # Preposition
    ]
    
    # Try to find word type at the beginning or after number
    for word_type in word_types:
        # Pattern: "1) Eylem" or just "Eylem" at start
        pattern = rf'(?:^\d+\s*\)\s*)?({word_type})\b'
        match = re.search(pattern, aciklama, re.IGNORECASE)
        if match:
            return word_type
    
    return "Bilinmiyor"


def normalize_gloss(gloss: str) -> str:
    """
    Normalize transcription (gloss) to standard format.
    - Convert to uppercase
    - Remove "Ceviri:" suffix and everything after
    - Handle special characters
    """
    if not gloss:
        return ""
    
    # Convert to uppercase
    gloss = gloss.upper()
    
    # Remove "Ceviri:" or "ÇEVİRİ:" and everything after (case-insensitive search on original)
    # Split on various possible spellings
    for separator in ["ÇEVİRİ:", "CEVIRI:", "CEVİRİ:", "CEVIRI"]:
        if separator in gloss:
            gloss = gloss.split(separator)[0]
            break
    
    # Remove newlines and normalize whitespace
    gloss = re.sub(r'[\n\r]+', ' ', gloss)
    gloss = normalize_whitespace(gloss)
    
    return gloss.strip()


def clean_sozluk_entry(data: Dict) -> Dict:
    """
    Clean a single TID_Sozluk_Verileri data.json entry.
    
    Args:
        data: Raw JSON data from data.json file
        
    Returns:
        Cleaned dictionary with standardized fields
    """
    kelime = (data.get("kelime") or "").strip()
    
    # Process all meanings
    anlamlar = data.get("anlamlar", [])
    cleaned_anlamlar = []
    
    for anlam in anlamlar:
        raw_aciklama = anlam.get("aciklama", "")
        
        # Clean the description
        aciklama = remove_html_tags(raw_aciklama)
        aciklama = remove_boilerplate(aciklama)
        aciklama = normalize_whitespace(aciklama)
        
        # Extract word type
        tur = extract_word_type(aciklama)
        
        # Get example if exists
        ornek = anlam.get("ornek", {})
        transkripsiyon = normalize_gloss(ornek.get("transkripsiyon", ""))
        ceviri = (ornek.get("ceviri") or "").strip()
        
        cleaned_anlam = {
            "sira_no": anlam.get("sira_no", 1),
            "tur": tur,
            "aciklama": aciklama,
            "transkripsiyon": transkripsiyon,
            "ceviri": ceviri,
        }
        cleaned_anlamlar.append(cleaned_anlam)
    
    # Deduplicate meanings (some entries have duplicates)
    seen = set()
    unique_anlamlar = []
    for anlam in cleaned_anlamlar:
        key = (anlam["aciklama"], anlam["transkripsiyon"])
        if key not in seen:
            seen.add(key)
            unique_anlamlar.append(anlam)
    
    return {
        "kelime": kelime,
        "ingilizce_karsiliklar": data.get("ingilizce_karsiliklar", []),
        "anlamlar": unique_anlamlar,
    }


def prepare_for_embedding(cleaned_entry: Dict) -> List[Dict]:
    """
    Prepare cleaned entry for ChromaDB embedding.
    Each meaning becomes a separate document.
    
    Returns:
        List of documents ready for embedding, each with:
        - document: Text to embed (the word itself)
        - metadata: All associated information
        - id: Unique identifier
    """
    kelime = cleaned_entry["kelime"]
    documents = []
    
    for i, anlam in enumerate(cleaned_entry["anlamlar"]):
        doc = {
            "document": kelime,  # The word itself is embedded
            "metadata": {
                "kelime": kelime,
                "tur": anlam["tur"],
                "aciklama": anlam["aciklama"],
                "ornek_transkripsiyon": anlam["transkripsiyon"],
                "ornek_ceviri": anlam["ceviri"],
                "anlam_index": i,
            },
            "id": f"{kelime.lower().replace(' ', '_')}_{i}",
        }
        documents.append(doc)
    
    return documents
