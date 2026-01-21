### TİD Sözlük Veri Çekme ve Medya İndirme - Staj Çalışması Raporu

Bu rapor, `https://tidsozluk.aile.gov.tr/tr/` adresindeki Güncel Türk İşaret Dili Sözlüğü'nden veri ve medya çekmek için geliştirilen `tid_scraper` aracının üç günlük staj çalışması boyunca nasıl planlandığını, geliştirildiğini ve çalıştırıldığını ayrıntılandırır. Rapor; literatür taraması, izlenen yöntem, mimari, kurulum ve çalıştırma adımları, kod alıntıları ve elde edilen sonuçlardan örnekler içerir.

---

### 1) Literatür Taraması ve Gereksinimler (Gün 1)

- **Amaç**: Dinamik web sitelerinden (SPA/MPA) veri çekme; sayfalandırma, içerik çıkarımı ve medya indirme.
- **Araçlar**: Selenium, webdriver-manager, requests, tqdm.
- **En İyi Uygulamalar**:
  - Headless tarayıcı ayarları: `--headless=new`, `--disable-dev-shm-usage`, `--no-sandbox`, sabit pencere boyutu.
  - Yüklenme bekleme: `document.readyState == 'complete'`, `WebDriverWait` ve `expected_conditions`.
  - Sayfalandırma: `rel="next"`, “Sonraki/İleri/»” tıklanabilirlik ve `disabled` durum kontrolü.
  - Dayanıklılık: Birden çok seçici stratejisi ve fallback; kelime bazında hata yalıtımı.
  - Etik/hukuki: Kullanım koşulları ve hız sınırlama; yalnızca eğitim/araştırma amaçlı kullanım.

---

### 2) Mimari Tasarım ve Veri Modeli (Gün 1)

- **Akış**: Harf linkleri → kelime URL’leri → kelime sayfası çıkarımı → dosya sistemi kaydı.
- **Sınıf**: `TIDScraper`
- **Veri modeli**: `WordData`, `MeaningItem`, `MeaningExample` (Python `@dataclass`).

Kod alıntıları:

```49:79:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/scraper.py
class TIDScraper:
    def __init__(
        self,
        output_directory: str = DEFAULT_OUTPUT_DIR,
        headless: bool = True,
        page_load_timeout_seconds: int = 20,
    ) -> None:
        self.base_url = BASE_URL
        self.output_directory = output_directory
        self.page_load_timeout_seconds = page_load_timeout_seconds
        self.driver = self._create_driver(headless=headless)
        os.makedirs(self.output_directory, exist_ok=True)
        logger.info("Çıktı klasörü: %s", os.path.abspath(self.output_directory))

    def _create_driver(self, headless: bool) -> webdriver.Chrome:
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--lang=tr-TR")
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(self.page_load_timeout_seconds)
        return driver
```

```118:132:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/scraper.py
def collect_all_word_urls(self, letters: Optional[List[str]] = None) -> List[str]:
    all_word_urls: Set[str] = set()
    letter_links = self.get_letter_links()

    if letters:
        letters_upper = {l.upper() for l in letters}
        letter_links = [u for u in letter_links if any(u.upper().endswith(f"/{l}") for l in letters_upper)]
        logger.info("Filtrelenmiş harf linkleri: %d", len(letter_links))

    for letter_url in letter_links:
        self._collect_letter_words(letter_url, all_word_urls)

    urls_sorted = sorted(all_word_urls)
    logger.info("Toplam kelime URL sayısı: %d", len(urls_sorted))
    return urls_sorted
```

```197:224:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/scraper.py
def process_word_page(self, word_url: str) -> WordData:
    self._get(word_url)
    page_html = self.driver.page_source

    derived = self._derive_word_from_url(word_url)
    word_name = self._extract_word_name_smart(derived)
    english_equivalents = self._extract_english_equivalents()
    video_url = self._extract_video_url(page_html)
    image_urls = self._extract_image_urls(page_html)
    meanings = self._extract_meanings()

    logger.debug(
        "Sayfa özet: word=%s, eng=%d, meanings=%d, video=%s, images=%d",
        word_name,
        len(english_equivalents),
        len(meanings),
        bool(video_url),
        len(image_urls),
    )

    return WordData(
        word=word_name,
        english_equivalents=english_equivalents,
        meanings=meanings,
        video_url=video_url,
        image_urls=image_urls,
    )
```

```393:417:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/scraper.py
def save_word(self, data: WordData) -> None:
    folder_name = self._sanitize_filename(data.word or "kelime") or "kelime"
    word_dir = os.path.join(self.output_directory, folder_name)
    images_dir = os.path.join(word_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    json_path = os.path.join(word_dir, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(self._to_json_dict(data), f, ensure_ascii=False, indent=4)
    logger.info("JSON kaydedildi: %s", json_path)

    if data.video_url:
        dest_video = os.path.join(word_dir, "video.mp4")
        self._download_file(data.video_url, dest_video)
        logger.info("Video indirildi: %s", dest_video)
    else:
        logger.warning("Video URL bulunamadı: %s", data.word)

    for idx, img_url in enumerate(data.image_urls, start=1):
        ext = os.path.splitext(urlparse(img_url).path)[1].lower() or ".jpg"
        filename = f"{idx:02d}{ext}"
        dest_img = os.path.join(images_dir, filename)
        self._download_file(img_url, dest_img)
        logger.debug("Görsel indirildi: %s", dest_img)
```

---

### 3) Kurulum ve Çalıştırma (Gün 2)

- Conda ortamı ve bağımlılıklar:

```12:18:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/README_TID_SCRAPER.md
pip install -r tid_scraper/requirements.txt
```

```1:5:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/requirements.txt
selenium>=4.22.0
webdriver-manager>=4.0.0
requests>=2.32.0
tqdm>=4.66.0
```

- CLI ile çalıştırma (tam sözlük veya sınırlı):

```24:38:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/README_TID_SCRAPER.md
python -m tid_scraper.main --headless
python -m tid_scraper.main --headless --letters A,Ç,G
python -m tid_scraper.main --headless --limit 50
```

```45:79:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/tid_scraper/main.py
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("tid_scraper.main")

    letters_list = None
    if args.letters:
        letters_list = [s.strip() for s in args.letters.split(",") if s.strip()]
        logger.info("Harf filtresi: %s", letters_list)

    scraper = TIDScraper(output_directory=args.output_dir, headless=args.headless)
    try:
        word_urls = scraper.collect_all_word_urls(letters=letters_list)
        logger.info("Toplam %d kelime URL bulundu", len(word_urls))
        if args.limit:
            word_urls = word_urls[: args.limit]
            logger.info("Limit uygulanıyor: %d kelime işlenecek", len(word_urls))

        for url in tqdm(word_urls, desc="Kelimeler işleniyor", unit="kelime"):
            try:
                logger.debug("İşleniyor: %s", url)
                data = scraper.process_word_page(url)
                scraper.save_word(data)
                logger.info("Kaydedildi: %s", data.word)
            except Exception as exc:
                logger.exception("Kelime işlenemedi: %s | Hata: %s", url, exc)
                continue
    finally:
        scraper.quit()

    return 0
```

---

### 4) Çalışma Sonuçları (Gün 3)

- Çıktı dizin yapısı: `@TID_Sozluk_Verileri/` altında her kelime için klasör, içinde `data.json`, varsa `video.mp4` ve `images/` klasörü.
- Örnek klasörler (kullanıcının paylaştığı): `İletişim`, `Örnek`, `Üniversite`, `İşaret`, `Öğrenci`, `Ülke`, vb.

Örnek `data.json` içerikleri:

```1:23:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/TID_Sozluk_Verileri/İletişim/data.json
{
    "kelime": "İletişim",
    "ingilizce_karsiliklar": [],
    "anlamlar": [
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "Güncel Türk İşaret Dili Sözlüğü\nSözlük Kullanımı\nHakkında\nProje Ekibi\nİletişim\nEN\nİletişim\nCommunication, Telecommunication, Conversation\n1) Ad Duygu, düşünce veya bilgilerin akla gelebilecek her türlü yolla başkalarına aktarılması, bildirişim, haberleşme, komünikasyon\nÖrnek :",
            "ornek": {
                "transkripsiyon": "BEN ANNE BABA KONUŞMAK BEN İLETİŞİM İMKANSIZ KESİLMEK\nÇeviri:\nAnne ve babamla konuşamıyorum, iletişim kurmamız imkansız.",
                "ceviri": "Anne ve babamla konuşamıyorum, iletişim kurmamız imkansız."
            }
        },
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "İletişim\nCommunication, Telecommunication, Conversation\n1) Ad Duygu, düşünce veya bilgilerin akla gelebilecek her türlü yolla başkalarına aktarılması, bildirişim, haberleşme, komünikasyon\nÖrnek :",
            "ornek": {
                "transkripsiyon": "BEN ANNE BABA KONUŞMAK BEN İLETİŞİM İMKANSIZ KESİLMEK\nÇeviri:\nAnne ve babamla konuşamıyorum, iletişim kurmamız imkansız.",
                "ceviri": "Anne ve babamla konuşamıyorum, iletişim kurmamız imkansız."
            }
        }
    ]
}
```

```1:23:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/TID_Sozluk_Verileri/Örnek/data.json
{
    "kelime": "Örnek",
    "ingilizce_karsiliklar": [],
    "anlamlar": [
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "Güncel Türk İşaret Dili Sözlüğü\nSözlük Kullanımı\nHakkında\nProje Ekibi\nİletişim\nEN\nÖrnek\nSpecimen, Example, Pattern, Type, Sample, Model, Duplication, Templete, Copy\n1) Ad Benzeri yapılacak olan, benzetilmek istenen şey, model\nÖrnek :",
            "ornek": {
                "transkripsiyon": "BEN RESİM FİKİR BULMAK^DEĞİL BİR ÖRNEK VERMEK BEN RESİM YAPMAK AYNI İSTEMEK BEN\nÇeviri:\nNe resmi yapacağımı bilmiyorum, bana bir örnek verirsen onu kopyalayarak yapabilirim.",
                "ceviri": "Ne resmi yapacağımı bilmiyorum, bana bir örnek verirsen onu kopyalayarak yapabilirim."
            }
        },
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "Örnek\nSpecimen, Example, Pattern, Type, Sample, Model, Duplication, Templete, Copy\n1) Ad Benzeri yapılacak olan, benzetilmek istenen şey, model\nÖrnek :",
            "ornek": {
                "transkripsiyon": "BEN RESİM FİKİR BULMAK^DEĞİL BİR ÖRNEK VERMEK BEN RESİM YAPMAK AYNI İSTEMEK BEN\nÇeviri:\nNe resmi yapacağımı bilmiyorum, bana bir örnek verirsen onu kopyalayarak yapabilirim.",
                "ceviri": "Ne resmi yapacağımı bilmiyorum, bana bir örnek verirsen onu kopyalayarak yapabilirim."
            }
        }
    ]
}
```

```1:24:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/TID_Sozluk_Verileri/Üniversite/data.json
{
    "kelime": "Üniversite",
    "ingilizce_karsiliklar": [],
    "anlamlar": [
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "Güncel Türk İşaret Dili Sözlüğü\nSözlük Kullanımı\nHakkında\nProje Ekibi\nİletişim\nEN\nÜniversite\nUniversity\n1) Ad Bilimsel özerkliğe ve kamu tüzel kişiliğine sahip, yüksek düzeyde eğitim, öğretim, bilimsel araştırma ve yayın yapan fakülte, enstitü, yüksekokul vb. kuruluş ve birimlerden oluşan öğretim kurumu\nÖrnek :",
            "ornek": {
                "transkripsiyon": "ÜNİVERSİTE SINAV KAZANMAK BEN GRAFİK RESİM OLMAK\nÇeviri:\nÜniversite sınavını kazandım, grafik-tasarım bölümünü okuyacağım.\nArama Sonuçlarına Dön",
                "ceviri": "Üniversite sınavını kazandım, grafik-tasarım bölümünü okuyacağım.\nArama Sonuçlarına Dön"
            }
        },
        {
            "sira_no": 1,
            "tur": "Ad",
            "aciklama": "Üniversite\nUniversity\n1) Ad Bilimsel özerkliğe ve kamu tüzel kişiliğine sahip, yüksek düzeyde eğitim, öğretim, bilimsel araştırma ve yayın yapan fakülte, enstitü, yüksekokul vb. kuruluş ve birimlerden oluşan öğretim kurumu\nÖrnek :",
            "ornek": {
                "transkripsiyon": "ÜNİVERSİTE SINAV KAZANMAK BEN GRAFİK RESİM OLMAK\nÇeviri:\nÜniversite sınavını kazandım, grafik-tasarım bölümünü okuyacağım.",
                "ceviri": "Üniversite sınavını kazandım, grafik-tasarım bölümünü okuyacağım."
            }
        }
    ]
}
```

- Gözlemler:
  - Bazı `data.json` dosyalarında anlam blokları yinelenebiliyor; bu, sayfa metninin tekrarlı bloklar içermesinden kaynaklı olabilir. Benzersizleştirme için `scraper.py` içinde anlam anahtarlarının birleştirilmesi uygulanıyor.
  - Video ve görseller her kelime için mevcut olmayabilir; kod bu durumda uyarı loglayıp devam eder.

---

### 5) Deneysel Ayarlar ve Senaryolar

- Sadece belirli harflerle çalıştırma: `--letters A,Ç,G`
- Hızlı debug: `--limit 50`
- Ayrıntılı log: `--verbose`
- Özel çıktı klasörü: `--output-dir /path/.../TID_Sozluk_Verileri`

---

### 6) Değerlendirme ve Gelecek Çalışmalar

- **Dayanıklılık**: Çoklu seçiciler, sayfalandırma fallback’leri ve kelime bazında hata yalıtımı ile süreç uzun koşularda stabil.
- **Veri kalitesi**: JSON yapısı standart; anlamlar için tekrarların filtrelenmesi geliştirilebilir.
- **İyileştirme önerileri**:
  - Oran sınırlama (rate limiting) ve yeniden deneme geri çekilme (exponential backoff).
  - Birim testler için HTML fixture’lar.
  - Değişen DOM’a karşı canary/sentinel kontroller.
  - İlerleme ve özet raporu için CSV/JSONL log dökümleri.

---

### 7) Sonuç

Bu çalışma ile TİD Sözlüğü’nden kelime bazlı metin ve medya içerikleri başarıyla çıkarılmış ve standart bir klasör yapısına kaydedilmiştir. Araç, uzun koşular için headless modda ve esnek parametrelerle kullanılabilir. Çıktılar, araştırma ve modelleme çalışmalarında (ör. veri kümesi hazırlama) doğrudan tüketilebilir niteliktedir.
