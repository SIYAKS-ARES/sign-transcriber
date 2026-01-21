import json
import os
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


logger = logging.getLogger("tid_scraper.scraper")

BASE_URL = "https://tidsozluk.aile.gov.tr/tr/"
DEFAULT_OUTPUT_DIR = "TID_Sozluk_Verileri"


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

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass

    def _get(self, url: str) -> None:
        logger.debug("GET %s", url)
        self.driver.get(url)
        WebDriverWait(self.driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    # ----------------------- URL TOPLAMA ----------------------- #
    def get_letter_links(self) -> List[str]:
        self._get(self.base_url)
        letter_links: Set[str] = set()

        anchors = self.driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                text = (a.text or "").strip()
            except WebDriverException:
                continue
            if not href:
                continue
            if re.search(r"/tr/Alfabetik/Arama/.+$", href):
                letter_links.add(href)
            elif len(text) in (1, 2) and re.search(r"/Alfabetik/Arama/", href):
                letter_links.add(href)

        if not letter_links:
            turkish_letters = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
            for letter in turkish_letters:
                letter_links.add(urljoin(self.base_url, f"Alfabetik/Arama/{letter}"))

        letter_links_sorted = sorted(letter_links)
        logger.info("%d harf linki bulundu", len(letter_links_sorted))
        return letter_links_sorted

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

    def _collect_letter_words(self, letter_url: str, word_url_bucket: Set[str]) -> None:
        next_selectors: List[Tuple[str, str]] = [
            (By.CSS_SELECTOR, "a[rel='next']"),
            (By.XPATH, "//a[contains(normalize-space(.), '\u00BB') or contains(., 'Sonraki') or contains(., 'İleri')]")
        ]

        self._get(letter_url)
        page_index = 1
        while True:
            page_source = self.driver.page_source
            hrefs = self._extract_word_hrefs_from_html(page_source)
            logger.info("%s [sayfa %d] -> %d kelime linki", letter_url, page_index, len(hrefs))
            for href in hrefs:
                absolute = urljoin(self.base_url, href)
                word_url_bucket.add(absolute)

            clicked_next = False
            for by, selector in next_selectors:
                try:
                    next_el = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((by, selector))
                    )
                except TimeoutException:
                    continue
                if not next_el:
                    continue
                classes = (next_el.get_attribute("class") or "").lower()
                aria_disabled = (next_el.get_attribute("aria-disabled") or "false").lower() == "true"
                if "disabled" in classes or aria_disabled:
                    continue
                current_url = self.driver.current_url
                try:
                    self.driver.execute_script("arguments[0].click();", next_el)
                except WebDriverException:
                    try:
                        next_el.click()
                    except Exception:
                        continue
                try:
                    WebDriverWait(self.driver, 10).until(EC.staleness_of(next_el))
                except TimeoutException:
                    pass
                try:
                    WebDriverWait(self.driver, 10).until(lambda d: d.current_url != current_url)
                except TimeoutException:
                    pass
                clicked_next = True
                page_index += 1
                break

            if not clicked_next:
                break

    @staticmethod
    def _extract_word_hrefs_from_html(html: str) -> Set[str]:
        # Daha geniş yakalama: hem href="/tr/... ?d=digit" hem de tam URL
        hrefs: Set[str] = set()
        for m in re.finditer(r'href="(/tr/[^"]+\?d=\d+)"', html):
            hrefs.add(m.group(1))
        for m in re.finditer(r'href="(https?://[^"]+/tr/[^"]+\?d=\d+)"', html):
            hrefs.add(m.group(1))
        return hrefs

    # ----------------------- KELİME SAYFASI ----------------------- #
    
    
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
        
    # ----------------------- KELİME ADI ÇIKARIMI ----------------------- #

    def _extract_word_name_smart(self, fallback_from_url: str) -> str:
        # h1/h2/h3 adaylarını topla ve site başlıklarını ele
        try:
            headers = self.driver.find_elements(By.XPATH, "//h1|//h2|//h3")
        except Exception:
            headers = []
        blacklist_substrings = ["Sözlük", "Sözlüğü", "Sözlük Kullanımı", "Hakkında", "Proje Ekibi", "İletişim", "EN"]
        candidates: List[str] = []
        for h in headers:
            try:
                txt = (h.text or "").strip()
            except Exception:
                continue
            if not txt:
                continue
            if any(b in txt for b in blacklist_substrings):
                continue
            if len(txt) > 40:
                continue
            # Harf ve Türkçe karakter, boşluk/dash izin ver
            if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü\-\s]+", txt):
                candidates.append(txt)
        if candidates:
            logger.debug("Kelime adı adayları: %s", candidates)
            return candidates[0]
        return fallback_from_url

    def _extract_english_equivalents(self) -> List[str]:
        candidates = self.driver.find_elements(By.XPATH, "//*[contains(translate(., 'İINGILCE', 'iingilce'), 'ingilizce')]")
        for el in candidates:
            text = (el.text or "").strip()
            if not text:
                continue
            if ":" in text:
                after_colon = text.split(":", 1)[1].strip()
                if after_colon:
                    parts = [p.strip() for p in after_colon.split(",") if p.strip()]
                    if parts:
                        return parts
        return []

    @staticmethod
    def _extract_video_url(html: str) -> Optional[str]:
        m = re.search(r"https?://[^'\"\s>]+\.mp4", html)
        if m:
            return m.group(0)
        m2 = re.search(r"src=\"(/[^\"]+\.mp4)\"", html)
        if m2:
            return urljoin(BASE_URL, m2.group(1))
        return None

    @staticmethod
    def _extract_image_urls(html: str) -> List[str]:
        srcs = set()
        for m in re.finditer(r"src=\"([^\"]+\.(?:jpg|jpeg|png))\"", html, re.IGNORECASE):
            url = m.group(1)
            if any(x in url.lower() for x in ["logo", "icon", "sprite", "favicon"]):
                continue
            if url.startswith("http"):
                srcs.add(url)
            else:
                srcs.add(urljoin(BASE_URL, url))
        return sorted(srcs)

    def _extract_meanings(self) -> List[MeaningItem]:
        meanings: List[MeaningItem] = []
        blocks = self.driver.find_elements(By.XPATH, "//*[contains(., 'TRANSKR') or contains(., 'Transkr') or contains(., 'Çeviri')]")
        visited = set()
        for el in blocks:
            container = self._nearest_container(el)
            if not container:
                container = el
            element_id = container.id if hasattr(container, 'id') else id(container)
            if element_id in visited:
                continue
            visited.add(element_id)
            text = (container.text or "").strip()
            if not text:
                continue

            order, kind = self._parse_order_and_kind(text)
            description = self._parse_description(text)
            transcription = self._parse_labeled_field(text, label_candidates=["TRANSKRİPSİYON", "TRANSKRIPSIYON", "Transkripsiyon"])
            translation = self._parse_labeled_field(text, label_candidates=["Çeviri", "CEVIRI", "Ceviri"])
            meanings.append(
                MeaningItem(
                    order=order,
                    kind=kind,
                    description=description,
                    example=MeaningExample(transcription=transcription, translation=translation),
                )
            )
        unique: List[MeaningItem] = []
        seen_keys = set()
        for mitem in meanings:
            key = (mitem.order, (mitem.kind or ""), (mitem.description or "")[:80], (mitem.example.transcription or "")[:80])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(mitem)
        return unique

    @staticmethod
    def _nearest_container(el):
        container = el
        for _ in range(8):
            try:
                tag = container.tag_name.lower()
            except Exception:
                break
            if tag == "html":
                break
            try:
                parent = container.find_element(By.XPATH, "./..")
            except Exception:
                break
            classes = (parent.get_attribute("class") or "").lower()
            ptag = parent.tag_name.lower()
            if any(k in classes for k in ["panel", "card", "anlam", "well", "group"]) or ptag in ("section", "article", "li", "div"):
                container = parent
            else:
                break
        return container

    @staticmethod
    def _parse_order_and_kind(text: str) -> Tuple[Optional[int], Optional[str]]:
        m = re.search(r"(\d+)[\)\.]\s*([A-Za-zÇĞİÖŞÜçğıöşü]+)?", text)
        if m:
            order = int(m.group(1))
            kind = (m.group(2) or "").strip() or None
            return order, kind
        return None, None

    @staticmethod
    def _parse_description(text: str) -> Optional[str]:
        cleaned = re.sub(r"TRANSKR[İI]PS[İI]YON.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"Çeviri.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*\d+[\)\.]\s*[A-Za-zÇĞİÖŞÜçğıöşü]*\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned[:400]
        return None

    @staticmethod
    def _parse_labeled_field(text: str, label_candidates: List[str]) -> Optional[str]:
        for label in label_candidates:
            pattern = re.compile(label + r"\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
            m = pattern.search(text)
            if m:
                value = m.group(1).strip()
                value = re.split(r"\n\s*\d+[\)\.]|\n\s*[A-ZÇĞİÖŞÜ]{3,}\s*:\s*", value)[0]
                return value.strip()
        return None

    @staticmethod
    def _derive_word_from_url(url: str) -> str:
        path = urlparse(url).path.rstrip("/").split("/")
        if path:
            return path[-1]
        return "kelime"

    # ----------------------- İNDİRME / KAYDETME ----------------------- #
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        allowed = "-_ .()" + "0123456789" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "çğıöşüÇĞİÖŞÜ"
        return "".join(ch if ch in allowed else "_" for ch in name)

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

    @staticmethod
    def _to_json_dict(data: WordData) -> Dict:
        return {
            "kelime": data.word,
            "ingilizce_karsiliklar": data.english_equivalents,
            "anlamlar": [
                {
                    "sira_no": m.order,
                    "tur": m.kind,
                    "aciklama": m.description,
                    "ornek": {
                        "transkripsiyon": m.example.transcription,
                        "ceviri": m.example.translation,
                    },
                }
                for m in data.meanings
            ],
        }

    @staticmethod
    def _download_file(url: str, destination_path: str, timeout_seconds: int = 30) -> None:
        try:
            if os.path.exists(destination_path):
                return
            with requests.get(url, stream=True, timeout=timeout_seconds) as r:
                r.raise_for_status()
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                with open(destination_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as exc:
            logger.warning("İndirme başarısız: %s -> %s | Hata: %s", url, destination_path, exc)
