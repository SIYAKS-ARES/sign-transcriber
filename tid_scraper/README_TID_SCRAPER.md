## TİD Sözlük Veri Çekme ve Medya İndirme Aracı

Bu araç, `https://tidsozluk.aile.gov.tr/tr/` adresindeki Güncel Türk İşaret Dili Sözlüğü'nden kelimeleri toplar, her kelimenin metinsel verilerini `data.json` olarak kaydeder ve ilişkili video ile görselleri indirir.

### Kurulum

1. Conda ortamını oluştur ve/veya etkinleştir:

```bash
conda env list | grep -w selenium >/dev/null 2>&1 || conda create -n selenium python=3.10 -y
conda activate selenium
```

2. Bağımlılıkları yükle:

```bash
pip install -r tid_scraper/requirements.txt
```

### Kullanım

Tam sözlüğü (A-Z) indirip `TID_Sozluk_Verileri/` klasörüne kaydetmek için:

```bash
python -m tid_scraper.main --headless
```

Sadece belirli harflerle başlamak için (ör. A, Ç, G):

```bash
python -m tid_scraper.main --headless --letters A,Ç,G
```

Debug amaçlı ilk 50 kelimeyle sınırlamak için:

```bash
python -m tid_scraper.main --headless --limit 50
```

Çıktı klasörünü değiştirmek için:

```bash
python -m tid_scraper.main --headless --output-dir /path/to/TID_Sozluk_Verileri
```

### Çıktı Yapısı

- `TID_Sozluk_Verileri/<Kelime>/data.json`
- `TID_Sozluk_Verileri/<Kelime>/video.mp4`
- `TID_Sozluk_Verileri/<Kelime>/images/01.jpg, 02.jpg, ...`

### Notlar

- Seçici ve DOM yapısı değişikliklerine dayanıklı olması için birden fazla strateji uygulanmıştır. Bazı sayfalarda beklenen alanlar yoksa alanlar boş bırakılır.
- Ağ veya sayfa bazlı hatalar tek kelime seviyesinde yutulur ve süreç devam eder.
- Headless modda çalıştırmak önerilir.
