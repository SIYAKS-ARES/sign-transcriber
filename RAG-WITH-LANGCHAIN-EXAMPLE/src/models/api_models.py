"""
API Request/Response modelleri
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .combination import CombinationFilters
from .question import GeneratedQuestion


class GenerateRequest(BaseModel):
    """
    Soru uretim istegi
    """

    filters: Optional[CombinationFilters] = Field(
        default=None, description="Kombinasyon filtreleri"
    )
    specific_combination: Optional[dict] = Field(
        default=None, description="Spesifik kombinasyon (filtre yerine)"
    )
    style_instruction: Optional[str] = Field(
        default=None, description="Stil talimati (cesitlilik icin)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filters": {
                    "alt_konu": "ebob_ekok",
                    "zorluk": 4,
                },
                "style_instruction": "Hikayede bir fabrika senaryosu kullan.",
            }
        }


class GenerateResponse(BaseModel):
    """
    Soru uretim yaniti
    """

    success: bool = Field(..., description="Basarili mi")
    data: Optional[GeneratedQuestion] = Field(default=None, description="Uretilen soru")
    error: Optional[str] = Field(default=None, description="Hata mesaji")
    raw_response: Optional[str] = Field(default=None, description="Ham LLM ciktisi (hata durumunda)")
    metadata: Optional[dict] = Field(default=None, description="Ek bilgiler")


class BatchGenerateRequest(BaseModel):
    """
    Toplu soru uretim istegi
    """

    count: int = Field(..., ge=1, le=50, description="Uretilecek soru sayisi")
    filters: Optional[CombinationFilters] = Field(
        default=None, description="Kombinasyon filtreleri"
    )
    ensure_diversity: bool = Field(
        default=True, description="Cesitlilik garantisi"
    )


class BatchGenerateResponse(BaseModel):
    """
    Toplu soru uretim yaniti
    """

    success: bool = Field(..., description="Basarili mi")
    data: Optional[dict] = Field(default=None, description="Sonuclar")
    error: Optional[str] = Field(default=None, description="Hata mesaji")


class HealthResponse(BaseModel):
    """
    Saglik kontrolu yaniti
    """

    status: str = Field(..., description="Sistem durumu")
    version: str = Field(..., description="API versiyonu")
    timestamp: datetime = Field(..., description="Zaman damgasi")
    components: dict = Field(default_factory=dict, description="Bilesen durumlari")


class ErrorResponse(BaseModel):
    """
    Hata yaniti
    """

    success: bool = Field(default=False)
    error: dict = Field(..., description="Hata detaylari")

