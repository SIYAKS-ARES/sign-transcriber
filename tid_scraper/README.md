# TID Scraper

Güncel Türk İşaret Dili Sözlüğü’nden (https://tidsozluk.aile.gov.tr) kelime, anlam, örnek ve medya (görsel/video) içeriği çeken Selenium tabanlı araç.

---

## Özellikler
- Harf bazlı gezinme ve tüm kelime URL’lerini toplama
- Her kelime sayfasından yapılandırılmış JSON çıkarma
- Görsel ve video dosyalarını indirip kelime klasörüne kaydetme
- Headless/visible tarayıcı modu
- Harf filtresi ve kelime limiti (debug) desteği

---

## Kurulum

```bash
cd tid_scraper
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Gereksinimler:
- Python 3.9+
- Chrome / Chromium (webdriver-manager otomatik indirir)

---

## Kullanım

### Komut satırı
```bash
python -m tid_scraper.main \
    --output-dir ../TID_Sozluk_Verileri \
    --letters A,Ç,G \
    --limit 50 \
    --headless \
    --verbose
```

Parametreler:
- `--output-dir`: Çıktı klasörü (varsayılan: `TID_Sozluk_Verileri`)
- `--letters`: Virgülle ayrılmış harf filtresi (ör: `A,Ç,G`)
- `--limit`: Maksimum kelime sayısı (debug)
- `--headless`: Headless tarayıcı
- `--verbose`: Ayrıntılı log

Çalıştırma sonrası her kelime için `output-dir/<Kelime>/` klasörü oluşur.

### Python içinden
```python
from tid_scraper.scraper import TIDScraper

scraper = TIDScraper(output_directory="TID_Sozluk_Verileri", headless=True)
urls = scraper.collect_all_word_urls(letters=["A", "B"])
for url in urls[:10]:
    data = scraper.process_word_page(url)
    scraper.save_word(data)
scraper.quit()
```

---

## Çıktı Yapısı

Her kelime için klasör:
```
TID_Sozluk_Verileri/<Kelime>/
  data.json       # Yapılandırılmış metin
  video.mp4       # (varsa) sözlük videosu
  images/         # (varsa) sözlük görselleri
```

`data.json` şeması (örnek):
```json
{
  "kelime": "Aç",
  "ingilizce_karsiliklar": [],
  "anlamlar": [
    {
      "sira_no": 1,
      "tur": "Sıfat",
      "aciklama": "...",
      "ornek": {
        "transkripsiyon": "...",
        "ceviri": "..."
      }
    }
  ],
  "video_url": "...",
  "image_urls": ["...", "..."]
}
```

---

## İpuçları ve Sorun Giderme
- **Rate limit / yavaş yükleme**: `--headless` kapatarak gerçek tarayıcı davranışını gözleyin; `page_load_timeout_seconds` (scraper.py) gerekirse artırılabilir.
- **Eksik medya**: Bazı kelimelerde video/görsel olmayabilir; scraper boş bırakır.
- **Chromium sürümü**: Webdriver-manager otomatik indirir; şirket ağlarında proxy gerekebilir.

---

## Lisans ve Katkı
Bu klasör, sözlük verisi çekme amaçlıdır. Çekilen verilerin telif/kullanım koşulları için resmi sözlük sitesinin şartlarını dikkate alın. Katkı ve iyileştirme için PR açabilirsiniz.
