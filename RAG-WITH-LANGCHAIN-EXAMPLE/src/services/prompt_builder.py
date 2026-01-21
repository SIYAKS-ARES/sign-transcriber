"""
R-05: Prompt Tasarimi
LGS tarzi soru uretimi icin system ve user prompt sablonlari
"""

from typing import Optional
import random


class PromptBuilder:
    """
    LGS soru uretimi icin prompt olusturucu
    
    Ozellikler:
    - LGS formatina uygun system prompt
    - Dinamik user prompt
    - Stil varyasyonlari (cesitlilik icin)
    - JSON cikti formati
    """

    def __init__(self):
        self.system_prompt = self._build_system_prompt()
        self._style_variations = self._get_style_variations()

    def _build_system_prompt(self) -> str:
        """Tam LGS sorusu (hikaye + soru + secenekler + cozum) icin system prompt"""
        return """Sen 8. sinif LGS matematik ogretmenisin.

Gorevin, verilen konu ve zorluk seviyesine uygun COK KALITELI bir LGS matematik sorusu olusturmaktir.

Kurallar:
- Soru mutlaka gercek hayata dayali KISA bir hikaye icerir.
- Hikayeden sonra tek bir coktan secmeli soru sor.
- En az 4 secenek ver (A, B, C, D). Istersen E secenegi de ekleyebilirsin.
- Sadece BIR dogru cevap olsun.
- Cozum adim adim, ogrenci anlayacak sekilde aciklanmalidir.
- Cevap formatin HER ZAMAN JSON olacak (hikaye, soru, secenekler, dogru_cevap, cozum alanlari)."""

    def _get_style_variations(self) -> list[str]:
        """Cesitlilik icin stil varyasyonlari"""
        return [
            "Hikayede bir ogrenci senaryosu kullan.",
            "Hikayede bir is yeri/fabrika senaryosu kullan.",
            "Hikayede bir spor/oyun senaryosu kullan.",
            "Hikayede bir aile/ev senaryosu kullan.",
            "Hikayede bir okul/sinif senaryosu kullan.",
            "Sayilari 100'den buyuk sec.",
            "Sayilari 50'den kucuk sec.",
            "Negatif bir durum iceren senaryo kullan (eksik, kayip, vb.).",
            "Karsilastirma iceren bir senaryo kullan.",
            "Zaman iceren bir senaryo kullan (gun, saat, dakika).",
            "Alisveris senaryosu kullan.",
            "Paylasim/bolme senaryosu kullan.",
            "Yarisma veya yarismaci senaryosu kullan.",
            "Toplama veya birlestirme senaryosu kullan.",
            "Seyahat veya yolculuk senaryosu kullan.",
        ]

    def build_user_prompt(
        self,
        combination: dict,
        formatted_examples: str,
        additional_instructions: Optional[str] = None,
    ) -> str:
        """Tam soru uretimi icin user prompt"""
        alt_konu = combination.get("alt_konu", "ebob_ekok")
        zorluk = combination.get("zorluk", 3)
        gorsel_tipi = combination.get("gorsel_tipi", "yok")

        extra = additional_instructions or ""

        # Not: formatted_examples RAG tarafindan hazirlansa da, token tasarrufu icin
        # burada gondermiyoruz. Sadece konu, zorluk ve stil talimati ile calisiyoruz.

        return f"""Konu: {alt_konu}
    Zorluk: {zorluk}/5
    Gorsel tipi: {gorsel_tipi}

    Gorevin:
- Bu konuya ve zorluga uygun, gercek hayat senaryolu bir hikaye yaz.
- Hikayeden sonra KONUYU ODAK ALAN tek bir coktan secmeli soru yaz.
- En az A, B, C, D olmak uzere 4 secenek yaz.
- Sadece bir dogru cevap olsun.
- Cozumu adim adim, aciklayici sekilde yaz.
- Eger gorsel_tipi "yok" degilse, gorselin nasil olacagini aciklayan KISA bir ifade yaz.

Ek stil talimati (opsiyonel, bos olabilir): {extra}

Sadece asagidaki JSON formatinda cevap ver. JSON disinda hicbir aciklama yazma:

{self._get_output_format()}"""

    def _get_output_format(self) -> str:
                """Tam soru JSON formati"""
                return """```json
{
    "hikaye": "Gercek hayat senaryolu, kisa matematik hikayesi.",
    "soru": "Hikayeye uygun coktan secmeli soru metni.",
    "gorsel_aciklama": "Eger varsa, gorselin kisaca aciklamasi (yok ise null).",
    "secenekler": {
        "A": "A secenegi metni",
        "B": "B secenegi metni",
        "C": "C secenegi metni",
        "D": "D secenegi metni"
    },
    "dogru_cevap": "A",
    "cozum": [
        "1. adim: ...",
        "2. adim: ...",
        "3. adim: ..."
    ],
    "kontroller": null
}
```"""

    def get_random_style(self, exclude_recent: int = 3, recent_styles: Optional[list[str]] = None) -> str:
        """
        Rastgele stil secimi
        
        Args:
            exclude_recent: Son N stili haric tut
            recent_styles: Son kullanilan stiller
            
        Returns:
            Stil talimati
        """
        recent_styles = recent_styles or []
        available = [s for s in self._style_variations if s not in recent_styles[-exclude_recent:]]

        if not available:
            available = self._style_variations

        return random.choice(available)

    def get_all_styles(self) -> list[str]:
        """Tum stil varyasyonlarini don"""
        return self._style_variations.copy()


# Hazir prompt sablonlari
PROMPT_TEMPLATES = {
    "default": PromptBuilder(),
}


def get_prompt_builder(template: str = "default") -> PromptBuilder:
    """
    Prompt builder instance al
    
    Args:
        template: Sablon adi
        
    Returns:
        PromptBuilder instance
    """
    if template not in PROMPT_TEMPLATES:
        PROMPT_TEMPLATES[template] = PromptBuilder()
    return PROMPT_TEMPLATES[template]

