import argparse
import logging
from typing import List, Optional

from tqdm import tqdm

from .scraper import TIDScraper, DEFAULT_OUTPUT_DIR


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Güncel TİD Sözlüğü'nden veri ve medya çekme aracı"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Çıktı klasörü (varsayılan: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--letters",
        default=None,
        help="Sadece belirtilen harflerle başla (virgülle ayrılmış, ör: A,Ç,G)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="İşlenecek maksimum kelime sayısı (debug için)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Tarayıcıyı headless modda çalıştır",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Ayrıntılı log çıktısı",
    )
    return parser.parse_args(argv)


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


if __name__ == "__main__":
    raise SystemExit(main())
