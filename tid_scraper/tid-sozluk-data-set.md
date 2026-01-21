# TİD Sözlük Veri Seti Özeti

Bu doküman, `tid_scraper` aracıyla `https://tidsozluk.aile.gov.tr/tr/` üzerindeki Güncel Türk İşaret Dili (TİD) Sözlüğü'nden çekilen `TID_Sozluk_Verileri/` veri setine dair tüm araştırmaları tek yerde toplar. Amaç; verinin kapsamını, yapısını, üretim sürecini ve olası kullanım senaryolarını belgeleyerek gelecekte gelen “Bu veri seti ne içeriyor?” sorularına hızlıca yanıt verebilmektir.

---

## 1. Yüksek Seviye Görünüm
- Kaynak: Aile ve Sosyal Hizmetler Bakanlığı TİD Sözlüğü (web tabanlı, alfabetik sayfalandırma).
- Toplama aracı: `tid_scraper` (Selenium + requests + tqdm) Python paketi.
- Çıktı dizini: `TID_Sozluk_Verileri/<Kelime>/` (her kelime için bağımsız klasör).
- Kapsam (20 Kasım 2025 itibarıyla): **1.933 kelime klasörü**, tamamında `data.json` mevcut, **2 kelimede video eksik**, tüm klasörlerde en az bir görsel var (hesaplama yerel doğrulama betiği ile yapılmıştır).
- Tipik içerik: metinsel tanımlar (`data.json`), kelimeye ait video (`video.mp4`) ve görseller (`images/01.jpg` vb.).

---

## 2. Veri Toplama Aracı (`tid_scraper`)

### 2.1 Mimari ve Veri Modeli
- `MeaningExample`, `MeaningItem` ve `WordData` `@dataclass` şemaları; kelimenin adı, İngilizce karşılıkları, anlam listesi, video ve görsel URL’lerini standartlaştırır.
- Selenium tabanlı `TIDScraper` sınıfı; harf bazlı gezinti, sayfalandırma, kelime sayfası işleme ve dosya sistemine yazma adımlarını kapsar.
- Çoklu seçici stratejileri (`rel="next"`, “Sonraki/İleri/»” vb.) ve fallback’ler, dinamik sayfa yapılarında kararlılık sağlar.

```26:132:tid_scraper/scraper.py
@dataclass
class MeaningExample:
    transcription: Optional[str]
    translation: Optional[str]

@dataclass
class MeaningItem:
    order: Optional[int]
    kind: Optional[str]
    description: Optional[str]
    example: MeaningExample

@dataclass
class WordData:
    word: str
    english_equivalents: List[str]
    meanings: List[MeaningItem]
    video_url: Optional[str]
    image_urls: List[str]

class TIDScraper:
    def __init__(...):
        self.base_url = BASE_URL
        self.output_directory = output_directory
        self.driver = self._create_driver(headless=headless)
...
    def collect_all_word_urls(...):
        letter_links = self.get_letter_links()
        ...
```

### 2.2 Kurulum ve Çalıştırma
- Önerilen ortam: `conda create -n selenium python=3.10` ve `pip install -r tid_scraper/requirements.txt`.
- Tam sözlüğü çekmek: `python -m tid_scraper.main --headless`.
- Seçili harfler: `--letters A,Ç,G`.
- Hızlı test: `--limit 50`.
- Çıktı konumu değiştirme: `--output-dir /hedef/pat` parametresi.

```1:51:tid_scraper/README_TID_SCRAPER.md
## TİD Sözlük Veri Çekme ve Medya İndirme Aracı
...
python -m tid_scraper.main --headless
python -m tid_scraper.main --headless --letters A,Ç,G
python -m tid_scraper.main --headless --limit 50
python -m tid_scraper.main --headless --output-dir /path/to/TID_Sozluk_Verileri
```

### 2.3 Geliştirme Notları (Staj Raporu)
- Dinamik siteler için literatür taraması; headless Chrome ayarları (`--headless=new`, `--disable-dev-shm-usage`, `--no-sandbox`), `WebDriverWait` ile `document.readyState == "complete"` kontrolleri.
- Etik kullanım: hız sınırlama, sadece eğitim/araştırma amaçlı kullanım vurgusu.
- Gelecek iyileştirmeleri: rate limiting, exponential backoff, DOM değişikliklerine karşı sentinel kontrolleri, HTML fixture’larıyla testler.
- Gözlemsel bulgu: bazı kelimelerde anlam blokları tekrarlanabiliyor; gerekirse anlam benzersizleştirme sonrası tüketilmeli.

```7:309:tid_scraper/tid-scraper-staj.md
- **Amaç**: Dinamik web sitelerinden veri çekme...
- Headless tarayıcı ayarları: `--headless=new`, `--disable-dev-shm-usage`, `--no-sandbox`.
- Yüklenme bekleme: `document.readyState == 'complete'`, `WebDriverWait`.
- Dayanıklılık: Çoklu seçici stratejisi, kelime bazında hata izolasyonu.
...
- Veri kalitesi: JSON yapısı standart; anlamlar için tekrarların filtrelenmesi geliştirilebilir.
- İyileştirme önerileri: rate limiting, exponential backoff, canary kontroller, CSV/JSONL özet logları.
```

---

## 3. Veri Dizini ve Dosya Yapısı

### 3.1 Klasör Hiyerarşisi
- Her kelime için `TID_Sozluk_Verileri/<Kelime>/` klasörü bulunur.
- Alt ögeler:
  - `data.json`: Metinsel içerik.
  - `video.mp4`: Kelimenin işaret videosu (2 kelimede eksik).
  - `images/`: Numaralı görseller (tüm kelimelerde en az bir görüntü var).
- Dosya adları Türkçe karakter içerir; dosya işlemlerinde UTF-8 ve path normalizasyonu önerilir.

```46:51:tid_scraper/README_TID_SCRAPER.md
- `TID_Sozluk_Verileri/<Kelime>/data.json`
- `TID_Sozluk_Verileri/<Kelime>/video.mp4`
- `TID_Sozluk_Verileri/<Kelime>/images/01.jpg, 02.jpg, ...`
```

### 3.2 `data.json` Şeması
- JSON dosyası her kelime için tekil bir kaydı temsil eder; aşağıdaki alanlar standarttır:

| Alan | Tip / Format | Notlar |
| --- | --- | --- |
| `kelime` | string | Klasör adıyla birebir uyumlu, Türkçe karakter içerebilir. |
| `ingilizce_karsiliklar` | string[] | İdeal olarak İngilizce sözcük listesi; bazı girdiler ham paragraf taşıyor (bkz. §3.3.1). |
| `anlamlar` | object[] | Her anlam için bağımsız blok; tekrar eden girdiler görülebilir. |
| `anlamlar[].sira_no` | int \| null | Tanım sırası; bazı girdilerde `null`. |
| `anlamlar[].tur` | string \| null | Dil bilgisi türü (`Ad`, `Eylem`, vb.). |
| `anlamlar[].aciklama` | string \| null | Tanım + sayfa başlıkları; satır sonları ve HTML kalıntıları içerir. |
| `anlamlar[].ornek.transkripsiyon` | string \| null | TİD gloss metni; `\n` karakterleri korunur. |
| `anlamlar[].ornek.ceviri` | string \| null | Türkçe doğal dil çevirisi. |

#### Standart Kayıt Örneği (`İletişim`)

```1:24:TID_Sozluk_Verileri/İletişim/data.json
{
    "kelime": "İletişim",
    "ingilizce_karsiliklar": [],
    "anlamlar": [
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "...",
            "ornek": {
                "transkripsiyon": "BEN ANNE BABA KONUŞMAK ...",
                "ceviri": "Anne ve babamla konuşamıyorum, iletişim kurmamız imkansız."
            }
        },
        ...
    ]
}
```

### 3.3 JSON Format Varyasyonları ve Karşılaştırmalar

#### 3.3.1 İngilizce Karşılıklar Alanındaki Gürültü
- Bazı kelimelerde `ingilizce_karsiliklar` alanı İngilizce sözcükler yerine HTML’den gelen ham paragraf metni içeriyor. Kaynak sayfadaki span/paragraf yapısı Selenium tarafından düz metne çevrilince örnek ve açıklamalar da bu alana karışıyor.

```1:30:TID_Sozluk_Verileri/Bilmek/data.json
"ingilizce_karsiliklar": [
    "TRANSKRİPSİYON:\nBİR KİTAP OKUMAK BAKMAK ...",
    "hatırlamak\nÖrnek :\nTRANSKRİPSİYON:\nBİRAZ SONRA TOPLANTI ...",
    ...
]
```

- **Temizleme Önerisi**: Regex ile yalnızca `[A-Za-z ,/-]+` desenini kabul eden token’ları saklamak veya sözlük sayfasından İngilizce karşılıkları yeniden parse etmek.

#### 3.3.2 Eksik / Null Alanlı Anlam Blokları
- Bazı anlam kayıtları tamamen `null` içeriyor:

```1:32:TID_Sozluk_Verileri/Çevirmek/data.json
{
    "sira_no": null,
    "tur": null,
    "aciklama": null,
    "ornek": {
        "transkripsiyon": null,
        "ceviri": null
    }
}
```

- **Temizleme Önerisi**: `aciklama` veya `ornek` alanı boş olan blokları filtreleyin; aksi halde inference pipeline’ında `None` kontrolleri gerekir.

#### 3.3.3 Tekrarlayan Anlamlar
- Aynı `aciklama` içeriği art arda tekrar edebiliyor (`Çevirmek`, `Açmak`, `Bilmek` vb.). `hash(aciklama)` veya `set` tabanlı benzersizleştirme ile bu tekrarlar kaldırılabilir.

### 3.4 Nicel Özet (20 Kasım 2025)
| Metrik | Değer |
| --- | --- |
| Toplam kelime klasörü | 1.933 |
| `data.json` eksik klasör | 0 |
| `video.mp4` eksik klasör | 2 |
| Görsel eksik klasör | 0 |

(Yukarıdaki değerler, depo kökünde çalıştırılan yardımcı Python betiğinden elde edilmiştir.)

---

## 4. Kalite Gözlemleri ve Kullanım Önerileri
- **Yinelenen anlam blokları**: Bazı sayfalarda tanım alanları tekrarlı dönebiliyor; modelleme öncesi benzersizleştirme/temizleme yapılmalı.
- **Eksik alanlar**: `null` içeren anlam bloklarını filtreleyin veya uygulama tarafında boş değer kontrolleri ekleyin.
- **İngilizce sütunu**: Gürültülü metni regex veya sözlük tabanlı filtrelerle temizleyin.
- **Medya tutarlılığı**: Video eksik kelimeler için alternatif kaynak aranmalı veya pipeline bu duruma göre hata toleranslı yazılmalı.
- **Metin temizliği**: `aciklama` alanları HTML kalıntıları veya satır sonu karakterleri içerebiliyor; normalizasyon (ör. `\n` temizleme) önerilir.
- **Karakter seti**: Tüm JSON’lar UTF-8 ve Türkçe karakter içerir; Python’da `ensure_ascii=False` kullanıldığı için downstream sistemler de UTF-8 kabul etmelidir.
- **Performans & Etik**: Uzun süreli scraping koşularında hız kısıtlama ve exponential backoff uygulanması beklenen iyi-pratik (bkz. staj raporu önerileri).

---

## 5. Örnek Kullanım Senaryoları
- **Kelime Bazlı Veri Keşfi**: `data.json` dosyaları, Türkçe-İngilizce karşılıklar ve örnek cümlelerle TİD söz varlığını incelemek için kullanılabilir.
- **Multimodal Model Eğitimi**: `video.mp4` + `images/` ile işaret dili tanıma, dudak/gest alt-tasks vb. için etiketli multimodal örnekler sunar.
- **Dil Teknolojileri**: `transkripsiyon` ve `ceviri` alanları sayesinde metin tabanlı çeviri veya gloss üretim modelleri eğitilebilir.
- **Soru-Cevap / RAG**: Tanım ve örnekleri JSON formatında sorgulamak, bilgi tabanlı uygulamalar veya RAG zincirleri için uygundur.

---

## 6. Gelecek Çalışmalar
- **İstatistiksel Raporlama**: Kategorilere göre (örn. `tur` alanı) dağılım çıkarma, kelime uzunluğu, örnek sayısı gibi metrikleri tabloya dökme.
- **Kalite İyileştirme**: Anlam tekrarlarını gideren post-processing, eksik video/görseller için yeniden deneme modülü.
- **Sürümleme**: `TID_Sozluk_Verileri` klasörünün tarih damgalı sürümlerini oluşturup değişim takibi yapmak.
- **Analitik Pipeline**: JSON verisini tekil `tid_sozluk.parquet` dosyasında toplamak; model eğitimleri için doğrudan veri seti sağlayıcı rolü.

---

### Hızlı Erişim
- Toplama aracı: `tid_scraper/`
- Veri deposu: `TID_Sozluk_Verileri/`
- Bu özet: `tid-sozluk-data-set.md`

