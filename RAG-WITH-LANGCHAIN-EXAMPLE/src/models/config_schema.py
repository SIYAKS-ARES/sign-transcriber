"""
configs.json semasi
Model katmanindan gelen konfigurasyon dosyasi
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .combination import Combination, SelectionPolicy


class ConfigSchema(BaseModel):
    """
    configs.json semasi (Model katmani ciktisi)
    """

    schema_version: str = Field(default="1.0", description="Sema versiyonu")
    model_version: str = Field(..., description="Model versiyonu")
    created_at: Optional[str] = Field(default=None, description="Olusturulma tarihi")
    threshold: float = Field(
        ..., ge=0, le=1, description="Minimum LGS skor esigi"
    )
    total_combinations: Optional[int] = Field(
        default=None, description="Toplam kombinasyon sayisi"
    )
    filtered_combinations: Optional[int] = Field(
        default=None, description="Threshold ustu kombinasyon sayisi"
    )
    selection_policy: Optional[SelectionPolicy] = Field(
        default=None, description="Kombinasyon secim politikasi"
    )
    combinations: list[Combination] = Field(
        ..., description="Kombinasyon listesi (sirali)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "schema_version": "1.0",
                "model_version": "lgs-struct-score-v1.0",
                "created_at": "2025-12-24",
                "threshold": 0.75,
                "total_combinations": 75,
                "filtered_combinations": 28,
                "selection_policy": {
                    "mode": "weighted_sampling",
                    "temperature": 1.0,
                    "alpha": 2.0,
                    "top_k": 20,
                },
                "combinations": [
                    {
                        "alt_konu": "ebob_ekok",
                        "zorluk": 4,
                        "gorsel_tipi": "sematik",
                        "lgs_skor": 0.92,
                        "rank": 1,
                    }
                ],
            }
        }

    @classmethod
    def create_mock(cls) -> "ConfigSchema":
        """
        Test/gelistirme icin mock config olustur
        Model katmani configs.json uretene kadar kullanilir
        """
        combinations = [
            # Yuksek skorlu kombinasyonlar
            Combination(
                alt_konu="ebob_ekok", zorluk=4, gorsel_tipi="sematik", lgs_skor=0.92, rank=1
            ),
            Combination(
                alt_konu="ebob_ekok", zorluk=3, gorsel_tipi="resimli", lgs_skor=0.89, rank=2
            ),
            Combination(
                alt_konu="carpanlar", zorluk=4, gorsel_tipi="tablo", lgs_skor=0.87, rank=3
            ),
            Combination(
                alt_konu="carpanlar", zorluk=3, gorsel_tipi="sematik", lgs_skor=0.85, rank=4
            ),
            Combination(
                alt_konu="aralarinda_asal", zorluk=4, gorsel_tipi="yok", lgs_skor=0.83, rank=5
            ),
            Combination(
                alt_konu="ebob_ekok", zorluk=5, gorsel_tipi="geometrik_sekil", lgs_skor=0.81, rank=6
            ),
            Combination(
                alt_konu="ebob_ekok", zorluk=2, gorsel_tipi="yok", lgs_skor=0.79, rank=7
            ),
            Combination(
                alt_konu="carpanlar", zorluk=2, gorsel_tipi="resimli", lgs_skor=0.77, rank=8
            ),
            Combination(
                alt_konu="aralarinda_asal", zorluk=3, gorsel_tipi="sematik", lgs_skor=0.76, rank=9
            ),
            Combination(
                alt_konu="carpanlar", zorluk=5, gorsel_tipi="yok", lgs_skor=0.75, rank=10
            ),
        ]

        return cls(
            schema_version="1.0",
            model_version="mock-v1.0",
            created_at=datetime.now().isoformat(),
            threshold=0.75,
            total_combinations=75,
            filtered_combinations=len(combinations),
            selection_policy=SelectionPolicy(),
            combinations=combinations,
        )

