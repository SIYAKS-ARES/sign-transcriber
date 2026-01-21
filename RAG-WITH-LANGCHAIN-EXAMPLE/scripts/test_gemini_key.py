"""Basit Gemini API key test script'i.

Bu script:
- Sizden Gemini API key'inizi ister (interaktif)
- Opsiyonel olarak model adini ve test prompt'unu alir
- Gemini'ye tek bir kucuk istek atar ve sonucu ya da hatayi yazdirir
"""

import sys


def main() -> None:
    # google-genai paketi kurulu mu kontrol et
    try:
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]
    except ImportError:
        print("✗ google-genai paketi kurulu degil!")
        print("Kurulum icin: pip install google-genai")
        sys.exit(1)

    print("Gemini API key testine hos geldiniz.\n")

    api_key = input("Gemini API key'inizi girin: ").strip()
    if not api_key:
        print("✗ API key girilmedi, cikiliyor.")
        sys.exit(1)

    default_model = "gemini-2.5-flash"
    model = input(f"Kullanilacak model adi [varsayilan: {default_model}]: ").strip() or default_model

    default_prompt = "Lutfen Turkce olarak 'API key calisiyor' yaz."
    prompt = input("Test prompt'u (bos birakirsen varsayilan kullanilir): ").strip() or default_prompt

    print("\n✓ Istek hazirlaniyor...\n")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:  # pragma: no cover - sadece manuel testte kullanilacak
        print("✗ Gemini client olusturulurken hata olustu:")
        print(repr(e))
        sys.exit(1)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=64,
            ),
        )

        # Cevabi metin olarak al
        text = getattr(response, "text", str(response))

        print("✓ Istek basarili!\n")
        print("Model:", model)
        print("Cevap (ilk 500 karakter):")
        print("-" * 60)
        print(text[:500])
        print("\n" + "-" * 60)

        # Varsa usage bilgisi goster
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            print("Kullanilan token bilgileri:")
            print(usage)

        sys.exit(0)

    except Exception as e:  # pragma: no cover - manuel test
        print("✗ Gemini istegi basarisiz oldu. Detay:")
        print(repr(e))
        print("\nBu hata mesajinda genelde su tip bilgiler olur:")
        print("  - 'UNAUTHENTICATED': API key gecersiz / yanlis\n"
              "  - 'PERMISSION_DENIED': Bu modeli kullanma izni yok\n"
              "  - 'RESOURCE_EXHAUSTED': Kota dolu / limit sifir\n"
              "  - 'UNAVAILABLE': Servis gecici olarak yanit veremiyor")
        sys.exit(1)


if __name__ == "__main__":
    main()
