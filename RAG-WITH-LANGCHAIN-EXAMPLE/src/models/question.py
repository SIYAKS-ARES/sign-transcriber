"""
Soru modelleri
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Question(BaseModel):
    """
    Veritabanindaki soru modeli
    """

    soru_metni: str = Field(..., description="Soru metni (OCR)")
    alt_konu: str = Field(..., description="Alt konu")
    zorluk: int = Field(..., ge=1, le=5, description="Zorluk seviyesi")
    gorsel_tipi: str = Field(..., description="Gorsel tipi")
    kaynak_tipi: str = Field(..., description="Kaynak tipi (cikmis/ornek/baslangic)")
    egitim_agirligi: float = Field(default=0.5, description="Egitim agirligi")
    is_lgs: int = Field(default=0, description="LGS profili mi (0/1)")
    ocr_kelime_sayisi: Optional[int] = Field(default=None)
    ocr_karakter_sayisi: Optional[int] = Field(default=None)
    ocr_rakam_sayisi: Optional[int] = Field(default=None)
    ocr_cok_adimli: Optional[int] = Field(default=None)


class RetrievedQuestion(BaseModel):
    """
    Retrieval sonucu soru modeli
    """

    soru_metni: str = Field(..., description="Soru metni")
    alt_konu: str = Field(..., description="Alt konu")
    zorluk: int = Field(..., description="Zorluk seviyesi")
    gorsel_tipi: str = Field(..., description="Gorsel tipi")
    kaynak_tipi: str = Field(..., description="Kaynak tipi")
    similarity_score: float = Field(default=0.0, description="Benzerlik skoru")
    metadata: dict = Field(default_factory=dict, description="Ek metadata")


class GeneratedQuestion(BaseModel):
    """
    Uretilen soru modeli (LLM ciktisi)
    """

    id: Optional[str] = Field(default=None, description="Benzersiz ID")
    alt_konu: str = Field(..., description="Alt konu")
    zorluk: int = Field(..., ge=1, le=5, description="Zorluk seviyesi")
    gorsel_tipi: str = Field(..., description="Gorsel tipi")
    hikaye: str = Field(..., description="Soru hikayesi/baglami")
    soru: str = Field(..., description="Asil soru metni")
    gorsel_aciklama: Optional[str] = Field(
        default=None, description="Gorsel aciklamasi"
    )
    secenekler: dict[str, str] = Field(..., description="A, B, C, D secenekleri")
    dogru_cevap: str = Field(..., pattern="^[ABCD]$", description="Dogru cevap")
    cozum: list[str] = Field(..., description="Cozum adimlari")
    kontroller: Optional[dict] = Field(default=None, description="Kalite kontrol sonuclari")
    metadata: Optional[dict] = Field(default=None, description="Uretim metadata")
    created_at: Optional[datetime] = Field(default=None, description="Olusturulma zamani")

    class Config:
        json_schema_extra = {
            "example": {
                "alt_konu": "ebob_ekok",
                "zorluk": 4,
                "gorsel_tipi": "sematik",
                "hikaye": "Bir fabrikada uretilen urunler kutulara...",
                "soru": "Buna gore, en az kac kutu gereklidir?",
                "gorsel_aciklama": "3 sutun 4 satirlik tablo",
                "secenekler": {"A": "12", "B": "15", "C": "18", "D": "24"},
                "dogru_cevap": "C",
                "cozum": [
                    "Adim 1: Urun miktarlarini belirleyelim",
                    "Adim 2: EBOB hesaplayalim",
                    "Adim 3: Toplam kutu sayisini bulalim",
                ],
            }
        }

