"""
Kombinasyon modelleri
(alt_konu, zorluk, gorsel_tipi) uclusu
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class Combination(BaseModel):
    """
    LGS soru kombinasyonu
    """

    alt_konu: Literal["carpanlar", "ebob_ekok", "aralarinda_asal"] = Field(
        ..., description="Alt konu kategorisi"
    )
    zorluk: int = Field(..., ge=1, le=5, description="Zorluk seviyesi (1-5)")
    gorsel_tipi: Literal["yok", "resimli", "geometrik_sekil", "sematik", "tablo"] = (
        Field(..., description="Gorsel tipi")
    )
    lgs_skor: float = Field(
        default=0.0, ge=0, le=1, description="LGS uygunluk skoru (0-1)"
    )
    rank: Optional[int] = Field(default=None, description="Skor siralamasi")

    class Config:
        json_schema_extra = {
            "example": {
                "alt_konu": "ebob_ekok",
                "zorluk": 4,
                "gorsel_tipi": "sematik",
                "lgs_skor": 0.87,
                "rank": 1,
            }
        }


class CombinationFilters(BaseModel):
    """
    Kombinasyon filtreleme kriterleri
    """

    alt_konu: Optional[str] = Field(default=None, description="Alt konu filtresi")
    zorluk: Optional[int] = Field(default=None, description="Zorluk filtresi")
    zorluk_min: Optional[int] = Field(default=None, ge=1, le=5)
    zorluk_max: Optional[int] = Field(default=None, ge=1, le=5)
    gorsel_tipi: Optional[str] = Field(default=None, description="Gorsel tipi filtresi")
    min_lgs_skor: Optional[float] = Field(
        default=None, ge=0, le=1, description="Minimum LGS skoru"
    )


class SelectionPolicy(BaseModel):
    """
    Kombinasyon secim politikasi
    """

    mode: Literal["weighted_sampling", "top_k", "random"] = Field(
        default="weighted_sampling", description="Secim modu"
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Softmax temperature"
    )
    alpha: float = Field(
        default=2.0, ge=0.5, le=5.0, description="Agirlik ust degeri (skor^alpha)"
    )
    top_k: int = Field(default=20, ge=1, description="Top-K secim sayisi")

