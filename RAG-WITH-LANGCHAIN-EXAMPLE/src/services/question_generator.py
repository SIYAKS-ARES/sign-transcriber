"""
R-06: Soru Uretim Pipeline
generate_question(config, examples) fonksiyonu ve tam pipeline
"""

import json
import uuid
from typing import Optional
from datetime import datetime

from ..models.question import GeneratedQuestion, RetrievedQuestion
from ..models.combination import Combination
from .retriever import QuestionRetriever
from .output_formatter import RAGOutputFormatter
from .prompt_builder import PromptBuilder
from .llm_client import BaseLLMClient, parse_llm_response
from tenacity import RetryError
from ..config import settings


class QuestionGenerationError(Exception):
    """Soru uretim hatasi"""
    pass


class ValidationError(Exception):
    """Validasyon hatasi"""
    pass


class InsufficientExamplesError(Exception):
    """Yetersiz ornek hatasi"""
    pass


class QuestionGenerator:
    """
    Tam soru uretim pipeline'i
    
    Akis:
    1. Benzer sorulari retrieve et
    2. Ornekleri formatla
    3. Prompt olustur
    4. LLM cagir
    5. Parse ve validate
    """

    def __init__(
        self,
        retriever: QuestionRetriever,
        formatter: RAGOutputFormatter,
        prompt_builder: PromptBuilder,
        llm_client: BaseLLMClient,
    ):
        """
        Args:
            retriever: Soru retrieval servisi
            formatter: RAG cikti formatlayici
            prompt_builder: Prompt olusturucu
            llm_client: LLM API client
        """
        self.retriever = retriever
        self.formatter = formatter
        self.prompt_builder = prompt_builder
        self.llm = llm_client

    async def generate_question(
        self,
        combination: dict,
        style_instruction: Optional[str] = None,
        max_attempts: int = 1,
    ) -> GeneratedQuestion:
        """
        Tek soru uret
        
        Args:
            combination: {alt_konu, zorluk, gorsel_tipi, lgs_skor}
            style_instruction: Stil talimati (cesitlilik icin)
            max_attempts: Maksimum deneme sayisi
            
        Returns:
            GeneratedQuestion instance
            
        Raises:
            QuestionGenerationError: Uretim basarisiz
            InsufficientExamplesError: Yeterli ornek yok
        """
        # Step 1: Retrieval
        examples = self.retriever.retrieve_examples(combination, top_k=5)

        if len(examples) < 3:
            raise InsufficientExamplesError(
                f"Yetersiz ornek: {len(examples)} bulundu, minimum 3 gerekli"
            )

        # Step 2: Format
        formatted_examples = self.formatter.format_examples(examples, combination)

        # Step 3: Prompt
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(
            combination=combination,
            formatted_examples=formatted_examples,
            additional_instructions=style_instruction,
        )

        # Step 4 & 5: Generate with retry
        last_error = None
        raw_llm_response = None
        for attempt in range(max_attempts):
            try:
                # Temperature: her denemede biraz artir
                temperature = 0.7 + (attempt * 0.1)

                # LLM cagri
                response = await self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                raw_llm_response = response  # Ham yaniti sakla

                # Parse JSON (tam soru formati bekleniyor)
                question_data = parse_llm_response(response)

                # Validate
                self._validate_question(question_data, combination)

                # Return
                return GeneratedQuestion(
                    id=str(uuid.uuid4()),
                    alt_konu=combination.get("alt_konu", ""),
                    zorluk=combination.get("zorluk", 3),
                    gorsel_tipi=combination.get("gorsel_tipi", "yok"),
                    hikaye=question_data.get("hikaye", ""),
                    soru=question_data.get("soru", ""),
                    gorsel_aciklama=question_data.get("gorsel_aciklama"),
                    secenekler=question_data.get("secenekler", {}),
                    dogru_cevap=question_data.get("dogru_cevap", "A"),
                    cozum=question_data.get("cozum", []),
                    kontroller=question_data.get("kontroller"),
                    metadata={
                        "attempt": attempt + 1,
                        "retrieval_count": len(examples),
                        "style_instruction": style_instruction,
                        "temperature": temperature,
                        "combination_lgs_skor": combination.get("lgs_skor"),
                    },
                    created_at=datetime.now(),
                )

            except json.JSONDecodeError as e:
                last_error = f"JSON parse hatasi: {e}"
            except ValidationError as e:
                last_error = f"Validasyon hatasi: {e}"
            except RetryError as e:
                # tenacity RetryError icindeki son hatayi cikart
                inner = e.last_attempt.exception()
                last_error = f"LLM retry hatasi: {inner!r}"
            except Exception as e:
                last_error = f"Beklenmeyen hata: {e}"

        error = QuestionGenerationError(
            f"Max {max_attempts} deneme sonrasi basarisiz. Son hata: {last_error}"
        )
        error.raw_response = raw_llm_response  # Ham yaniti exception'a ekle
        raise error

    def _validate_question(self, data: dict, combination: dict):
        """Tam soru validasyonu"""
        # Zorunlu alanlar
        hikaye = data.get("hikaye", "").strip()
        soru = data.get("soru", "").strip()
        secenekler = data.get("secenekler", {}) or {}
        dogru_cevap = str(data.get("dogru_cevap", "")).strip()
        cozum = data.get("cozum", []) or []

        if not hikaye:
            raise ValidationError("Eksik alan: hikaye")

        if not soru:
            raise ValidationError("Eksik alan: soru")

        if not isinstance(secenekler, dict) or len(secenekler) < 4:
            raise ValidationError("Eksik veya hatali alan: secenekler (en az A,B,C,D olmali)")

        # Standart secenek harfleri
        required_options = {"A", "B", "C", "D"}
        if not required_options.issubset(set(secenekler.keys())):
            raise ValidationError("Secenekler A, B, C, D icermeli")

        if not dogru_cevap or dogru_cevap not in secenekler:
            raise ValidationError("dogru_cevap, secenekler'den biri olmali")

        if not isinstance(cozum, list) or len(cozum) == 0:
            raise ValidationError("Eksik alan: cozum (en az bir adim olmali)")

    async def generate_batch(
        self,
        combinations: list[dict],
        ensure_diversity: bool = True,
        max_per_combination: int = 1,
    ) -> list[GeneratedQuestion]:
        """
        Toplu soru uretimi
        
        Args:
            combinations: Kombinasyon listesi
            ensure_diversity: Cesitlilik garantisi
            max_per_combination: Kombinasyon basina maksimum soru
            
        Returns:
            GeneratedQuestion listesi
        """
        results = []
        used_styles = []

        for combo in combinations:
            for _ in range(max_per_combination):
                try:
                    # Cesitlilik icin rastgele stil
                    style = None
                    if ensure_diversity:
                        style = self.prompt_builder.get_random_style(
                            exclude_recent=3, recent_styles=used_styles
                        )
                        used_styles.append(style)

                    question = await self.generate_question(
                        combination=combo,
                        style_instruction=style,
                    )
                    results.append(question)

                except (QuestionGenerationError, InsufficientExamplesError) as e:
                    # Hatali kombinasyonu atla, devam et
                    print(f"Kombinasyon atlandi: {combo}. Hata: {e}")
                    continue

        return results


async def generate_single_question(
    retriever: QuestionRetriever,
    llm_client: BaseLLMClient,
    combination: dict,
    style_instruction: Optional[str] = None,
) -> GeneratedQuestion:
    """
    Basit tek soru uretimi (factory fonksiyonu)
    
    Args:
        retriever: Retriever instance
        llm_client: LLM client instance
        combination: Hedef kombinasyon
        style_instruction: Stil talimati
        
    Returns:
        GeneratedQuestion
    """
    generator = QuestionGenerator(
        retriever=retriever,
        formatter=RAGOutputFormatter(format_type="markdown"),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
    )

    return await generator.generate_question(
        combination=combination,
        style_instruction=style_instruction,
    )

