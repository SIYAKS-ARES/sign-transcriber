from .database import TIDVectorDB

class RAGEngine:
    def __init__(self):
        self.db = TIDVectorDB()

    def prepare_rag_context(self, gloss_sentence: str) -> str:
        """
        Retrieves context from both Memory (Level 1) and Dictionary (Level 2)
        and formats it into a prompt string.
        """
        context_parts = []
        
        # --- Level 1: Memory Retrieval (Few-Shot Examples) ---
        context_parts.append("--- REFERANS ÇEVİRİLER (Benzer Örnekler) ---")
        
        memory_results = self.db.query_memory(gloss_sentence, n_results=2)
        
        if memory_results['documents'] and memory_results['documents'][0]:
            for i, (doc, meta) in enumerate(zip(memory_results['documents'][0], memory_results['metadatas'][0])):
                # doc is the stored gloss sentence
                # meta contains 'dogru_ceviri'
                translation = meta.get('dogru_ceviri', 'Çeviri yok')
                context_parts.append(f"Örnek {i+1}: '{doc}' -> '{translation}'")
        else:
            context_parts.append("Daha önce benzer bir çeviri veri tabanında bulunamadı.")

        context_parts.append("") # Newline

        # --- Level 2: Dictionary Retrieval (Word Definitions) ---
        context_parts.append("--- SÖZLÜK BİLGİSİ (Kelime Detayları) ---")
        
        words = gloss_sentence.split()
        for word in words:
            # Basic cleanup for the word if needed (currently assuming clean gloss input)
            clean_word = word.strip()
            
            dict_results = self.db.query_word(clean_word, n_results=1)
            
            if dict_results['documents'] and dict_results['documents'][0]:
                found_word = dict_results['documents'][0][0]
                meta = dict_results['metadatas'][0][0]
                anlam_ozeti = meta.get('ozet_anlam', 'Anlam yok')
                
                # Check distance/relevance if needed, but for now we trust vector match
                context_parts.append(f"Kelime: '{clean_word}' (Eşleşen: '{found_word}')\n  -> Anlam: {anlam_ozeti}")
            else:
                context_parts.append(f"Kelime: '{clean_word}' -> Sözlükte bulunamadı.")

        return "\n".join(context_parts)

    def construct_final_prompt(self, gloss_sentence: str, context: str) -> str:
        """
        Combines the user input and the retrieved context into the final LLM prompt.
        """
        return f"""Sen uzman bir Türk İşaret Dili (TİD) çevirmenisin.
Aşağıdaki TİD gloss transkripsiyonunu, verilen bağlam bilgilerini kullanarak kurallı ve doğal bir Türkçe cümleye çevir.

GÖREV KURALLARI:
1. REFERANS ÇEVİRİLER kısmındaki örnekleri incele ve benzer kalıpları kullan.
2. SÖZLÜK BİLGİSİ kısmındaki kelime anlamlarını (İsim/Eylem ayrımını) dikkate al.
3. Asla halüsinasyon görme, sadece verilen kelimelerin ima ettiği anlamı çıkar.
4. Çıktı sadece Türkçe çeviri cümlesi olsun.

VERİLER (Bağlam):
{context}

GİRDİ (Transkripsiyon):
{gloss_sentence}

ÇIKTI (Türkçe Çeviri):
"""
