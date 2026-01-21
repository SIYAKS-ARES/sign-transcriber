"""
R-07: Cesitlilik (Randomness) Mekanizmasi
Ayni konfigurasyondan farkli sorular uretmek icin
"""

import random
from typing import Optional
from collections import deque

from .prompt_builder import PromptBuilder


class DiversityService:
    """
    Soru uretiminde cesitlilik saglayan servis
    
    Ozellikler:
    - Rastgele stil secimi
    - Dinamik temperature ayarlama
    - Tekrar tespiti (Jaccard similarity)
    - Ornek siralama karistirma
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        max_history: int = 100,
        duplicate_threshold: float = 0.85,
    ):
        """
        Args:
            prompt_builder: Prompt olusturucu (stil varyasyonlari icin)
            max_history: Gecmis soru sayisi limiti
            duplicate_threshold: Tekrar tespit esigi (0-1)
        """
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_history = max_history
        self.duplicate_threshold = duplicate_threshold

        # Gecmis takibi
        self.used_styles: deque[str] = deque(maxlen=20)
        self.recent_questions: deque[str] = deque(maxlen=max_history)
        self.recent_embeddings: deque[set[str]] = deque(maxlen=max_history)

    def get_random_style(self, exclude_recent: int = 3) -> str:
        """
        Rastgele stil secimi
        
        Args:
            exclude_recent: Son N stili haric tut
            
        Returns:
            Stil talimati string
        """
        all_styles = self.prompt_builder.get_all_styles()

        # Son kullanilanlari haric tut
        recent = list(self.used_styles)[-exclude_recent:] if self.used_styles else []
        available = [s for s in all_styles if s not in recent]

        if not available:
            available = all_styles

        selected = random.choice(available)
        self.used_styles.append(selected)

        return selected

    def get_temperature(
        self,
        attempt: int = 1,
        base: float = 0.7,
        variation: float = 0.1,
    ) -> float:
        """
        Dinamik temperature hesapla
        
        Args:
            attempt: Deneme sayisi (1-based)
            base: Baslangic temperature
            variation: Her denemede artis miktari
            
        Returns:
            Temperature degeri (0.5-1.0 arasi)
        """
        # Her denemede biraz artir
        temp = base + (attempt - 1) * variation

        # Rastgele kucuk varyasyon ekle
        temp += random.uniform(-0.05, 0.05)

        # Sinirlarda tut
        return max(0.5, min(1.0, temp))

    def add_prompt_variation(self, base_prompt: str) -> str:
        """
        Prompt'a rastgele varyasyon ekle
        
        Args:
            base_prompt: Temel prompt
            
        Returns:
            Varyasyonlu prompt
        """
        variations = [
            "\n\nNot: Sayilari orijinal tut, hikayeyi degistir.",
            "\n\nNot: Farkli bir senaryo kullan.",
            "\n\nNot: Daha kisa ve oz bir hikaye yaz.",
            "\n\nNot: Daha detayli bir hikaye yaz.",
            "\n\nNot: Gunluk hayattan ornek kullan.",
            "\n\nNot: Secenekleri daha yakin degerlerle olustur.",
            "\n\nNot: Cozumu daha detayli acikla.",
        ]

        return base_prompt + random.choice(variations)

    def is_duplicate(self, new_question: str) -> bool:
        """
        Tekrar kontrolu (Jaccard similarity)
        
        Args:
            new_question: Yeni soru metni
            
        Returns:
            True ise tekrar, False ise yeni
        """
        if not new_question:
            return False

        # Kelime seti olustur
        new_words = self._tokenize(new_question)

        # Son sorularla karsilastir
        for recent_words in self.recent_embeddings:
            similarity = self._jaccard_similarity(new_words, recent_words)
            if similarity > self.duplicate_threshold:
                return True

        return False

    def _tokenize(self, text: str) -> set[str]:
        """Metni kelime setine cevir"""
        # Basit tokenizasyon: kucuk harf, boslukla ayir
        words = text.lower().split()
        # Stop words'leri haric tutmuyoruz (daha iyi benzerlik icin)
        return set(words)

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Jaccard similarity hesapla"""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def add_to_history(self, question: str):
        """
        Soru gecmisine ekle
        
        Args:
            question: Soru metni
        """
        if question:
            self.recent_questions.append(question)
            self.recent_embeddings.append(self._tokenize(question))

    def shuffle_examples(
        self,
        examples: list,
        keep_first: int = 2,
    ) -> list:
        """
        Orneklerin sirasini karistir (ilk N haric)
        
        Args:
            examples: Ornek listesi
            keep_first: Ilk N tanesini yerinde tut (en iyi ornekler)
            
        Returns:
            Karistirilmis liste
        """
        if len(examples) <= keep_first:
            return examples

        fixed = examples[:keep_first]
        to_shuffle = examples[keep_first:]
        random.shuffle(to_shuffle)

        return fixed + to_shuffle

    def get_diverse_combinations(
        self,
        combinations: list[dict],
        count: int,
        prefer_high_score: bool = True,
    ) -> list[dict]:
        """
        Cesitli kombinasyon secimi
        
        Args:
            combinations: Tum kombinasyonlar
            count: Secilecek sayi
            prefer_high_score: Yuksek skorlulari tercih et
            
        Returns:
            Secilen kombinasyonlar
        """
        if len(combinations) <= count:
            return combinations.copy()

        # Skorlara gore agirlikli secim
        if prefer_high_score:
            weights = [c.get("lgs_skor", 0.5) ** 2 for c in combinations]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = None
        else:
            weights = None

        # Rastgele secim (tekrarsiz)
        indices = list(range(len(combinations)))
        selected_indices = []

        for _ in range(count):
            if not indices:
                break

            if weights:
                remaining_weights = [weights[i] for i in indices]
                total = sum(remaining_weights)
                if total > 0:
                    remaining_weights = [w / total for w in remaining_weights]
                    idx = random.choices(indices, weights=remaining_weights, k=1)[0]
                else:
                    idx = random.choice(indices)
            else:
                idx = random.choice(indices)

            selected_indices.append(idx)
            indices.remove(idx)

        return [combinations[i] for i in selected_indices]

    def clear_history(self):
        """Gecmisi temizle"""
        self.used_styles.clear()
        self.recent_questions.clear()
        self.recent_embeddings.clear()

    def get_stats(self) -> dict:
        """Istatistikleri don"""
        return {
            "used_styles_count": len(self.used_styles),
            "recent_questions_count": len(self.recent_questions),
            "max_history": self.max_history,
            "duplicate_threshold": self.duplicate_threshold,
        }


class DiverseQuestionGenerator:
    """
    Cesitlilik garantili soru uretici wrapper
    """

    def __init__(
        self,
        generator,  # QuestionGenerator
        diversity: DiversityService,
    ):
        """
        Args:
            generator: QuestionGenerator instance
            diversity: DiversityService instance
        """
        self.generator = generator
        self.diversity = diversity

    async def generate_diverse_batch(
        self,
        combination: dict,
        count: int = 5,
        max_duplicates: int = 3,
    ) -> list:
        """
        Cesitlilik garantili toplu uretim
        
        Args:
            combination: Hedef kombinasyon
            count: Uretilecek soru sayisi
            max_duplicates: Maksimum tekrar denemesi
            
        Returns:
            GeneratedQuestion listesi
        """
        results = []
        duplicate_count = 0

        while len(results) < count:
            # Rastgele stil
            style = self.diversity.get_random_style()

            try:
                # Uret
                question = await self.generator.generate_question(
                    combination=combination,
                    style_instruction=style,
                )

                # Tekrar kontrolu
                full_text = f"{question.hikaye} {question.soru}"
                if self.diversity.is_duplicate(full_text):
                    duplicate_count += 1
                    if duplicate_count >= max_duplicates:
                        # Cok fazla tekrar, daha yuksek temperature dene
                        print(f"Tekrar limiti asildi, temperature artiriliyor")
                        duplicate_count = 0
                    continue

                # Basarili
                self.diversity.add_to_history(full_text)
                results.append(question)
                duplicate_count = 0  # Reset

            except Exception as e:
                print(f"Uretim hatasi: {e}")
                duplicate_count += 1
                if duplicate_count >= max_duplicates * 2:
                    # Cok fazla hata, cikmis
                    break

        return results

