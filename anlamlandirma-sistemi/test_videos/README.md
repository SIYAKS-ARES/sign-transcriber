# Test Videos Rehberi (`anlamlandirma-sistemi/test_videos/`)

Bu klasör, `anlamlandirma-sistemi` için **örnek işaret dili videolarını** içerir.  
Temel amaç, model + RAG + LLM akışını hızlıca test etmek ve demo ekranları için standart örnekler sağlamaktır.

---

## İçerik

| Dosya | Açıklama |
|-------|----------|
| `acikmak_1.mp4`–`acikmak_5.mp4` | \"ACIKMAK\" işaretine ait farklı örnek videolar (farklı kişi/poz) |

Tüm videolar:
- Yaklaşık benzer süre ve kare sayısına sahiptir (MediaPipe + Transformer pipeline’ı ile uyumlu)
- Demo ve `test_transformer_model.py` script’i için kullanılır

---

## Nerede Kullanılıyor?

1. **Model Test Script’i**

```bash
cd anlamlandirma-sistemi
python test_transformer_model.a.py   # veya ilgili script
```

Script tipik olarak:
- `test_videos/acikmak_*.mp4` dosyalarını okur
- MediaPipe ile keypoint çıkarır
- Transformer modelinden top-k tahminleri ve güven skorlarını yazdırır

2. **Web Demo (`/demo`)**

- Tarayıcıda `http://localhost:5005/demo` adresine gidin
- “Video Yükle” bölümünden bu klasördeki videolardan birini seçin
- **Başlat** / **Test Model** / **Translate** butonları ile:
  - Sadece model tahmini,
  - veya model + LLM çevirisini test edin.

Detaylı adımlar için:
- `DEMO_CALISTIRMA.md`
- `HIZLI_TEST.md`

---

## Kendi Videolarınızı Eklemek

Ekstra test videoları eklerken:
- Dosya formatı: `.mp4` (H.264 önerilir)
- Çözünürlük: 720p civarı (veya Transformer eğitiminde kullanılan çözünürlüğe yakın)
- Süre: 1–3 saniye (yaklaşık 200 frame @ ~30 FPS)

Önerilen isimlendirme:

```
<kelime>_<index>.mp4
ör: selam_1.mp4, selam_2.mp4
```

Bu sayede test script’lerde basit wildcard ile (`glob('test_videos/selam_*.mp4')`) otomatik tarama yapılabilir.

---

**Son Güncelleme:** 2026-01-21

