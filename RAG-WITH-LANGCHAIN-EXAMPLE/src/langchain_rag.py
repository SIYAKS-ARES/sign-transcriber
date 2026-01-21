"""
LGS RAG with LangChain
Basit ve etkili RAG pipeline
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .config import settings


class LangChainRAG:
    """
    LangChain tabanli RAG sistemi
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        RAG sistemini baslat

        Args:
            openai_api_key: OpenAI API key (opsiyonel, .env'den alinir)
        """
        api_key = openai_api_key or settings.openai_api_key

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=api_key,
        )

        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vectorstore = None
        self.qa_chain = None

    def load_questions(self, csv_path: Optional[str] = None) -> int:
        """
        CSV'den sorulari yukle ve vectorstore olustur

        Args:
            csv_path: CSV dosya yolu (opsiyonel)

        Returns:
            Yuklenen soru sayisi
        """
        path = Path(csv_path) if csv_path else settings.questions_csv_path

        if not path.exists():
            raise FileNotFoundError(f"CSV dosyasi bulunamadi: {path}")

        df = pd.read_csv(path)

        # Dokumanlari olustur
        documents = []
        for _, row in df.iterrows():
            # Metin icerigi
            content = f"""
Soru: {row.get('Soru_MetniOCR', '')}
Alt Konu: {row.get('Alt_Konu', '')}
Zorluk: {row.get('Zorluk', '')}
Gorsel Tipi: {row.get('Gorsel_Tipi', '')}
Kaynak: {row.get('Kaynak_Tipi', '')}
            """.strip()

            # Metadata
            metadata = {
                "alt_konu": row.get("Alt_Konu", ""),
                "zorluk": int(row.get("Zorluk", 0)),
                "gorsel_tipi": row.get("Gorsel_Tipi", ""),
                "kaynak_tipi": row.get("Kaynak_Tipi", ""),
                "is_lgs": int(row.get("is_LGS", 0)),
            }

            documents.append(Document(page_content=content, metadata=metadata))

        # FAISS vectorstore olustur
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        print(f"Yuklendi: {len(documents)} soru")
        return len(documents)

    def save_vectorstore(self, path: Optional[str] = None):
        """Vectorstore'u kaydet"""
        save_path = Path(path) if path else settings.vector_store_path
        save_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(save_path))
        print(f"Vectorstore kaydedildi: {save_path}")

    def load_vectorstore(self, path: Optional[str] = None):
        """Vectorstore'u yukle"""
        load_path = Path(path) if path else settings.vector_store_path
        self.vectorstore = FAISS.load_local(
            str(load_path), self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"Vectorstore yuklendi: {load_path}")

    def create_qa_chain(self):
        """
        Soru-cevap zinciri olustur
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore yuklenmemis. Once load_questions() cagirin.")

        # LGS soru uretim promptu
        prompt_template = """
Sen LGS (Liseye Gecis Sinavi) matematik sorulari ureten uzman bir egitimcisin.

Asagidaki ornek sorulari referans alarak YENI ve OZGUN bir LGS sorusu uret.

ORNEK SORULAR:
{context}

HEDEF KOMBINASYON: {question}

KURALLAR:
1. Orneklerden FARKLI, tamamen yeni bir soru uret
2. 4 secenek (A, B, C, D) olmali
3. Tek dogru cevap olmali
4. Cozum adimlari acik ve anlasilir olmali
5. Zorluk seviyesine uygun ol

JSON formatinda yanit ver:
{{
    "hikaye": "Soru baglami/hikayesi",
    "soru": "Asil soru metni",
    "gorsel_aciklama": "Gorsel gerekliyse aciklama, degilse null",
    "secenekler": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "dogru_cevap": "A/B/C/D",
    "cozum": ["Adim 1...", "Adim 2...", "Adim 3..."]
}}

SADECE JSON CIKTISI VER, BASKA ACIKLAMA EKLEME.
"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": settings.retrieval_top_k}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        print("QA Chain olusturuldu")

    def generate_question(self, combination: dict) -> dict:
        """
        Kombinasyona gore soru uret

        Args:
            combination: {alt_konu, zorluk, gorsel_tipi}

        Returns:
            Uretilen soru (dict)
        """
        if self.qa_chain is None:
            raise ValueError("QA Chain olusturulmamis.Once create_qa_chain() cagirin.")

        # Query olustur
        query = f"Alt Konu: {combination.get('alt_konu', '')}, Zorluk: {combination.get('zorluk', '')}, Gorsel Tipi: {combination.get('gorsel_tipi', '')}"

        # Calistir
        result = self.qa_chain.invoke({"query": query})

        # JSON parse
        try:
            # Check if result has expected keys
            if "result" not in result:
                return {
                    "error": "LangChain QA chain sonucu beklenen formatta degil",
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "not_a_dict",
                    "combination": combination,
                }
            
            response_text = result["result"]
            # ```json ... ``` bloklarini temizle
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end == -1:
                    # Closing backticks missing, use rest of string
                    response_text = response_text[start:].strip()
                else:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end == -1:
                    # Closing backticks missing, use rest of string
                    response_text = response_text[start:].strip()
                else:
                    response_text = response_text[start:end].strip()

            question_data = json.loads(response_text.strip())
            question_data["combination"] = combination
            question_data["source_documents"] = len(result.get("source_documents", []))

            return question_data

        except (json.JSONDecodeError, KeyError) as e:
            return {
                "error": f"Parse hatasi: {e}",
                "raw_response": result.get("result", "N/A"),
                "combination": combination,
            }


# CLI kullanimi icin
if __name__ == "__main__":
    # Test
    rag = LangChainRAG()

    # Sorulari yukle
    rag.load_questions()

    # Chain olustur
    rag.create_qa_chain()

    # Ornek soru uret
    question = rag.generate_question(
        {"alt_konu": "ebob_ekok", "zorluk": 4, "gorsel_tipi": "sematik"}
    )

    print(json.dumps(question, ensure_ascii=False, indent=2))

