import sys
import uuid
from .database import TIDVectorDB
from .rag_engine import RAGEngine

def main():
    print("=== TID RAG Çeviri Sistemi Başlatılıyor ===")
    
    # Initialize Engine
    try:
        engine = RAGEngine()
        db = TIDVectorDB()
    except Exception as e:
        print(f"Hata: Sistem başlatılamadı. Detay: {e}")
        return

    print("Sistem Hazır. Çıkmak için 'q' veya 'exit' yazın.")
    print("-" * 50)

    while True:
        try:
            gloss_input = input("\nTranskripsiyon (Gloss) Girin: ").strip()
            
            if gloss_input.lower() in ['q', 'exit']:
                print("Çıkış yapılıyor...")
                break
            
            if not gloss_input:
                continue

            print("\n... Bağlam aranıyor ...")
            
            # 1. Retrieve & Build Context
            context = engine.prepare_rag_context(gloss_input)
            
            # 2. Construct Prompt
            final_prompt = engine.construct_final_prompt(gloss_input, context)
            
            print("\n" + "="*20 + " OLUŞTURULAN PROMPT " + "="*20)
            print(final_prompt)
            print("="*60)

            # 3. Simulate LLM Generation & Feedback Loop
            print("\n[LLM Entegrasyonu olmadığı için bu adım manueldir]")
            sys.stdout.write("Lütfen ideal çeviriyi girin (Geri besleme mekanizmasını test etmek için): ")
            sys.stdout.flush()
            
            user_translation = input()
            
            if user_translation:
                # 4. Save to Memory (Feedback Loop)
                print(f"\nBu çeviri hafızaya kaydedilsin mi? (e/h)")
                confirm = input().lower()
                
                if confirm == 'e':
                    doc_id = f"sent_{uuid.uuid4().hex[:8]}"
                    db.add_sentence_memory(gloss_input, user_translation, doc_id)
                    print("✅ Hafızaya kaydedildi! Bir sonraki benzer sorguda referans olarak kullanılacak.")
                else:
                    print("❌ Kaydedilmedi.")
            
        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.")
            break
        except Exception as e:
            print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
