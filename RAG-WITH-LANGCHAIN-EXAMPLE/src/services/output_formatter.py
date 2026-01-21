"""
R-04: RAG Ciktisinin Formatlenmasi
LLM'e verilecek ornek soru paketini temiz formatta hazirlama
"""

import json
from typing import Literal

from ..models.question import RetrievedQuestion


class RAGOutputFormatter:
    """
    Retrieval sonuclarini LLM icin formatlama
    
    Desteklenen formatlar:
    - markdown: En okunabilir, LLM icin ideal
    - json: Yapisal veri transferi icin
    - text: Basit duz metin
    """

    def __init__(self, format_type: Literal["markdown", "json", "text"] = "markdown"):
        """
        Args:
            format_type: Cikti formati
        """
        self.format_type = format_type

    def format_examples(
        self,
        examples: list[RetrievedQuestion],
        combination: dict,
    ) -> str:
        """
        Ornekleri LLM icin formatla
        
        Args:
            examples: Retrieve edilen sorular
            combination: Hedef kombinasyon
            
        Returns:
            Formatli string
        """
        if self.format_type == "json":
            return self._format_as_json(examples, combination)
        elif self.format_type == "markdown":
            return self._format_as_markdown(examples, combination)
        else:
            return self._format_as_text(examples, combination)

    def _format_as_markdown(
        self,
        examples: list[RetrievedQuestion],
        combination: dict,
    ) -> str:
        """Markdown formatinda cikti"""
        output = []

        # Hedef kombinasyon
        output.append("## Hedef Kombinasyon")
        output.append(f"- **Alt Konu:** {combination.get('alt_konu', 'N/A')}")
        output.append(f"- **Zorluk:** {combination.get('zorluk', 'N/A')}/5")
        output.append(f"- **Gorsel Tipi:** {combination.get('gorsel_tipi', 'N/A')}")
        if "lgs_skor" in combination:
            output.append(f"- **LGS Skoru:** {combination.get('lgs_skor', 'N/A')}")
        output.append("")
        output.append("## Ornek Sorular")
        output.append("")

        # Ornekler
        for i, ex in enumerate(examples, 1):
            output.append(f"### Ornek {i}")
            output.append(f"**Kaynak:** {ex.kaynak_tipi.upper()}")
            output.append(f"**Zorluk:** {ex.zorluk}/5")
            output.append(f"**Gorsel:** {ex.gorsel_tipi}")
            output.append("")
            output.append("**Soru:**")
            output.append(f"> {ex.soru_metni}")
            output.append("")
            output.append("---")
            output.append("")

        return "\n".join(output)

    def _format_as_json(
        self,
        examples: list[RetrievedQuestion],
        combination: dict,
    ) -> str:
        """JSON formatinda cikti"""
        data = {
            "hedef_kombinasyon": {
                "alt_konu": combination.get("alt_konu"),
                "zorluk": combination.get("zorluk"),
                "gorsel_tipi": combination.get("gorsel_tipi"),
                "lgs_skor": combination.get("lgs_skor"),
            },
            "ornek_sayisi": len(examples),
            "ornek_sorular": [
                {
                    "soru_metni": ex.soru_metni,
                    "alt_konu": ex.alt_konu,
                    "zorluk": ex.zorluk,
                    "gorsel_tipi": ex.gorsel_tipi,
                    "kaynak_tipi": ex.kaynak_tipi,
                }
                for ex in examples
            ],
        }

        return json.dumps(data, ensure_ascii=False, indent=2)

    def _format_as_text(
        self,
        examples: list[RetrievedQuestion],
        combination: dict,
    ) -> str:
        """Duz text formatinda cikti"""
        lines = []

        # Hedef
        lines.append(
            f"HEDEF: {combination.get('alt_konu', 'N/A')} | "
            f"Zorluk {combination.get('zorluk', 'N/A')} | "
            f"{combination.get('gorsel_tipi', 'N/A')}"
        )
        lines.append("")
        lines.append("ORNEK SORULAR:")
        lines.append("=" * 60)

        # Ornekler
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n[{i}] ({ex.kaynak_tipi}, Zorluk:{ex.zorluk})")
            lines.append(ex.soru_metni)
            lines.append("-" * 40)

        return "\n".join(lines)

    def format_for_context(self, examples: list[RetrievedQuestion]) -> str:
        """
        Sadece ornek sorulari context olarak formatla
        (Hedef kombinasyon olmadan)
        """
        if not examples:
            return "Ornek soru bulunamadi."

        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"[Ornek {i} - {ex.kaynak_tipi.upper()}, Z:{ex.zorluk}]")
            lines.append(ex.soru_metni)
            lines.append("")

        return "\n".join(lines)


def format_question_output(question_data: dict) -> str:
    """
    Uretilen soruyu guzel formatta goster
    
    Args:
        question_data: Uretilen soru dict
        
    Returns:
        Formatli string
    """
    lines = []

    lines.append("=" * 60)
    lines.append("URETILEN SORU")
    lines.append("=" * 60)
    lines.append("")

    # Metadata
    lines.append(f"Alt Konu: {question_data.get('alt_konu', 'N/A')}")
    lines.append(f"Zorluk: {question_data.get('zorluk', 'N/A')}/5")
    lines.append(f"Gorsel Tipi: {question_data.get('gorsel_tipi', 'N/A')}")
    lines.append("")

    # Hikaye
    lines.append("HIKAYE:")
    lines.append("-" * 40)
    lines.append(question_data.get("hikaye", ""))
    lines.append("")

    # Soru
    lines.append("SORU:")
    lines.append("-" * 40)
    lines.append(question_data.get("soru", ""))
    lines.append("")

    # Gorsel (varsa)
    if question_data.get("gorsel_aciklama"):
        lines.append("GORSEL ACIKLAMASI:")
        lines.append("-" * 40)
        lines.append(question_data.get("gorsel_aciklama", ""))
        lines.append("")

    # Secenekler
    lines.append("SECENEKLER:")
    lines.append("-" * 40)
    secenekler = question_data.get("secenekler", {})
    for key in ["A", "B", "C", "D"]:
        if key in secenekler:
            lines.append(f"  {key}) {secenekler[key]}")
    lines.append("")

    # Dogru cevap
    lines.append(f"DOGRU CEVAP: {question_data.get('dogru_cevap', 'N/A')}")
    lines.append("")

    # Cozum
    lines.append("COZUM:")
    lines.append("-" * 40)
    for i, adim in enumerate(question_data.get("cozum", []), 1):
        lines.append(f"  {adim}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)

