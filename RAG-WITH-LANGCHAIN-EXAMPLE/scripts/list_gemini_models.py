"""
Gemini API'den mevcut modelleri listele
"""
import os
from pathlib import Path
import sys

# .env yukleme
from dotenv import load_dotenv
load_dotenv()

# google-genai paketi kurulu mu kontrol et
try:
    from google import genai
    print("✓ google-genai paketi kurulu")
except ImportError:
    print("✗ google-genai paketi kurulu değil!")
    print("Kurulum için: pip install google-genai")
    sys.exit(1)

# API key kontrol
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("✗ GEMINI_API_KEY bulunamadı!")
    sys.exit(1)

print(f"✓ API Key bulundu: {api_key[:20]}...")

# Client olustur
try:
    client = genai.Client(api_key=api_key)
    print("✓ Gemini client başarıyla oluşturuldu\n")
except Exception as e:
    print(f"✗ Client oluşturulamadı: {e}")
    sys.exit(1)

# Modelleri listele
print("=" * 60)
print("MEVCUT GEMİNİ MODELLERİ")
print("=" * 60)

try:
    models = client.models.list()
    
    text_generation_models = []
    for model in models:
        # generateContent destekleyen modelleri filtrele
        if hasattr(model, 'supported_generation_methods'):
            if 'generateContent' in model.supported_generation_methods:
                text_generation_models.append(model)
        else:
            # Eski API formatı
            text_generation_models.append(model)
    
    if text_generation_models:
        print(f"\nText generation için kullanılabilir modeller ({len(text_generation_models)} adet):\n")
        for model in text_generation_models:
            name = model.name if hasattr(model, 'name') else str(model)
            print(f"  • {name}")
            
            # Ek bilgiler varsa göster
            if hasattr(model, 'display_name') and model.display_name:
                print(f"    Display Name: {model.display_name}")
            if hasattr(model, 'description') and model.description:
                desc = model.description[:100] + "..." if len(model.description) > 100 else model.description
                print(f"    Description: {desc}")
            print()
    else:
        print("Hiç text generation modeli bulunamadı!")
        print("\nTüm modeller:")
        for model in models:
            print(f"  • {model}")

except Exception as e:
    print(f"✗ Model listesi alınamadı: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("\nÖNERİLEN MODEL ADLARI (.env içinde kullanılabilir):")
print("  - gemini-2.0-flash-exp")
print("  - gemini-1.5-flash")
print("  - gemini-1.5-pro")
print("=" * 60)
