"""
TID RAG Translation System - Streamlit Dashboard
=================================================
Human-in-the-Loop feedback interface for thesis demonstration.

Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st

from retriever.dual_retriever import DualRetriever
from prompt_builder.augmented_prompt import AugmentedPromptBuilder
from feedback.feedback_handler import FeedbackHandler
from integration.input_adapter import adapt_input
from pipeline import TranslationPipeline


# Initialize components (cached)
@st.cache_resource
def get_retriever():
    return DualRetriever()


@st.cache_resource
def get_prompt_builder():
    return AugmentedPromptBuilder()


@st.cache_resource
def get_feedback_handler():
    return FeedbackHandler()


@st.cache_resource
def get_translation_pipeline():
    """Initialize translation pipeline (cached)."""
    return TranslationPipeline(provider="gemini")


def main():
    st.set_page_config(
        page_title="TID RAG Ceviri Sistemi",
        page_icon="🤟",
        layout="wide",
    )
    
    st.title("TID RAG Translation System")
    st.subheader("Human-in-the-Loop Feedback Interface")
    
    # Sidebar - System Statistics
    with st.sidebar:
        st.header("Sistem Istatistikleri")
        
        retriever = get_retriever()
        stats = retriever.get_stats()
        
        st.metric("Sozluk Kayit Sayisi", stats["sozluk_count"])
        st.metric("Hafiza Kayit Sayisi", stats["hafiza_count"])
        
        st.divider()
        
        st.header("Hakkinda")
        st.markdown("""
        Bu sistem, Turk Isaret Dili (TID) transkripsiyonlarini
        Turkce'ye cevirmek icin RAG (Retrieval-Augmented Generation)
        teknolojisi kullanir.
        
        **Ozellikler:**
        - Dual-Collection Strategy
        - Human-in-the-Loop Learning
        - BLEU/BERTScore Evaluation
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("1. Girdi")
        
        # Input field
        gloss_input = st.text_input(
            "Transkripsiyon (Gloss):",
            value="OKUL GITMEK ISTEMEK",
            help="TID transkripsiyonunu buyuk harflerle girin",
        )
        
        translate_btn = st.button("Cevir", type="primary", use_container_width=True)
    
    # Process translation
    if translate_btn and gloss_input:
        retriever = get_retriever()
        prompt_builder = get_prompt_builder()
        
        # Normalize input
        try:
            input_data = adapt_input(gloss_input)
            normalized_input = input_data.raw_string
        except Exception as e:
            st.error(f"Girdi hatasi: {e}")
            return
        
        # Get retrieval results
        with st.spinner("Veritabani sorgulanıyor..."):
            retrieval_result = retriever.retrieve(normalized_input)
        
        with col2:
            st.header("2. Bulunan Baglamsal Bilgi")
            
            # Memory results
            st.subheader("Hafiza (Benzer Cumleler)")
            if retrieval_result.similar_translations:
                for i, trans in enumerate(retrieval_result.similar_translations, 1):
                    with st.expander(f"Ornek {i} (Benzerlik: {trans['similarity']:.2f})"):
                        st.markdown(f"**Transkripsiyon:** {trans['transkripsiyon']}")
                        st.markdown(f"**Ceviri:** {trans['ceviri']}")
            else:
                st.info("Benzer ceviri bulunamadi.")
            
            # Dictionary results
            st.subheader("Sozluk (Kelime Bilgileri)")
            if retrieval_result.has_word_info():
                for word, infos in retrieval_result.word_info.items():
                    if infos:
                        info = infos[0]
                        metadata = info.get("metadata", {})
                        with st.expander(f"{word} - {metadata.get('tur', '?')}"):
                            st.markdown(f"**Aciklama:** {metadata.get('aciklama', '-')[:200]}...")
                            if metadata.get('ornek_ceviri'):
                                st.markdown(f"**Ornek:** {metadata.get('ornek_ceviri')}")
            else:
                st.info("Sozlukte eslesen kelime bulunamadi.")
        
        # Show generated prompt
        st.header("3. Olusturulan Prompt")
        prompt = prompt_builder.build_prompt(normalized_input)
        
        with st.expander("Prompt'u Goster", expanded=False):
            st.code(prompt, language="markdown")
        
        st.info(f"Prompt uzunlugu: {len(prompt)} karakter")
        
        # LLM Translation
        st.header("4. LLM Cevirisi")
        
        # Store translation in session state
        if "llm_translation" not in st.session_state:
            st.session_state.llm_translation = None
            st.session_state.llm_alternatives = []
            st.session_state.llm_confidence = 0
        
        # Translate button
        if st.button("LLM ile Cevir", type="secondary", use_container_width=True):
            with st.spinner("LLM cevirisi yapiliyor..."):
                try:
                    pipeline = get_translation_pipeline()
                    result = pipeline.translate(normalized_input)
                    
                    st.session_state.llm_translation = result.best_translation
                    st.session_state.llm_confidence = result.confidence
                    
                    if result.translation_result and result.translation_result.alternatives:
                        st.session_state.llm_alternatives = [
                            {
                                "translation": alt.translation,
                                "confidence": alt.confidence,
                                "explanation": alt.explanation
                            }
                            for alt in result.translation_result.alternatives
                        ]
                except Exception as e:
                    st.error(f"LLM hatasi: {e}")
        
        # Show LLM results
        if st.session_state.llm_translation:
            st.success(f"En iyi ceviri: **{st.session_state.llm_translation}** (Guven: {st.session_state.llm_confidence}/10)")
            
            if st.session_state.llm_alternatives:
                with st.expander("Tum Alternatifler"):
                    for i, alt in enumerate(st.session_state.llm_alternatives, 1):
                        st.markdown(f"**{i}.** [{alt['confidence']}/10] {alt['translation']}")
                        if alt.get('explanation'):
                            st.caption(alt['explanation'][:150] + "...")
        
        # Manual override or edit
        manual_translation = st.text_area(
            "Ceviri (duzenle veya manuel gir):",
            value=st.session_state.llm_translation or "",
            placeholder="Ceviriyi buraya girin...",
        )
        
        # Feedback section
        st.header("5. Geri Besleme")
        
        col_approve, col_reject = st.columns(2)
        
        with col_approve:
            if st.button("Onayla ve Hafizaya Kaydet", type="primary", use_container_width=True):
                if manual_translation:
                    feedback_handler = get_feedback_handler()
                    doc_id = feedback_handler.add_direct_translation(
                        transkripsiyon=normalized_input,
                        ceviri=manual_translation,
                        provider="manual",
                    )
                    st.success(f"Hafizaya kaydedildi! (ID: {doc_id})")
                    # Clear cache to update stats
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Lutfen ondan ceviri girin.")
        
        with col_reject:
            if st.button("Reddet", use_container_width=True):
                st.info("Ceviri reddedildi. Hafizaya kaydedilmedi.")
    
    # Search existing translations
    st.divider()
    st.header("Hafiza Arama")
    
    search_query = st.text_input("Aranacak transkripsiyon:", "")
    
    if search_query:
        retriever = get_retriever()
        results = retriever.hafiza.query(search_query, n_results=5)
        
        if results:
            for r in results:
                with st.expander(f"{r['transkripsiyon'][:50]}... (Benzerlik: {r['similarity']:.2f})"):
                    st.markdown(f"**Ceviri:** {r['ceviri']}")
                    st.markdown(f"**Provider:** {r['provider']}")
        else:
            st.info("Sonuc bulunamadi.")


if __name__ == "__main__":
    main()
