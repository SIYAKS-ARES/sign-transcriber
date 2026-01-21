"""
Config Loader Service
Loads configs.json and questions CSV for RAG system
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_int(value: str | None, default: int = 0) -> int:
    """
    Safely convert string to int, handling empty strings and None.
    
    Args:
        value: String value to convert (may be None or empty)
        default: Default value if conversion fails or value is empty
        
    Returns:
        Integer value or default
    """
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Could not convert '{value}' to int, using default {default}")
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    """
    Safely convert string to float, handling empty strings and None.
    
    Args:
        value: String value to convert (may be None or empty)
        default: Default value if conversion fails or value is empty
        
    Returns:
        Float value or default
    """
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Could not convert '{value}' to float, using default {default}")
        return default


class ConfigLoader:
    """
    Configuration and data loader for RAG system.
    Handles loading of:
    - configs.json (model output with scored combinations)
    - questions CSV (training/example questions)
    """

    def __init__(self):
        """Initialize ConfigLoader"""
        self._configs_cache: dict | None = None
        self._questions_cache: list[dict] | None = None

    def load_configs(self, path: str | Path) -> dict[str, Any]:
        """
        Load configs.json file containing scored combinations.

        Args:
            path: Path to configs.json file

        Returns:
            Dictionary containing:
            - schema_version: str
            - model_version: str
            - threshold: float
            - combinations: list[dict]
            - selection_policy: dict

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configs file not found: {path}")

        logger.info(f"Loading configs from: {path}")

        with open(path, "r", encoding="utf-8") as f:
            configs = json.load(f)

        # Validate required fields
        required_fields = ["combinations"]
        for field in required_fields:
            if field not in configs:
                raise ValueError(f"Missing required field in configs: {field}")

        # Set defaults
        configs.setdefault("threshold", 0.70)
        configs.setdefault("schema_version", "1.0")
        configs.setdefault("selection_policy", {
            "mode": "weighted_sampling",
            "temperature": 1.0,
            "alpha": 2.0
        })

        logger.info(
            f"Loaded {len(configs['combinations'])} combinations "
            f"(threshold: {configs['threshold']})"
        )

        self._configs_cache = configs
        return configs

    def load_questions(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load questions from CSV file.

        Args:
            path: Path to questions CSV file

        Returns:
            List of question dictionaries with fields:
            - Soru_MetniOCR: str
            - Alt_Konu: str
            - Zorluk: int
            - Gorsel_Tipi: str
            - Kaynak_Tipi: str
            - is_LGS: int
            - egitim_agirligi: float

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")

        logger.info(f"Loading questions from: {path}")

        questions = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields with safe conversion for empty strings
                question = {
                    "Soru_MetniOCR": row.get("Soru_MetniOCR", ""),
                    "Alt_Konu": row.get("Alt_Konu", ""),
                    "Zorluk": _safe_int(row.get("Zorluk", ""), default=0),
                    "Gorsel_Tipi": row.get("Gorsel_Tipi", "yok"),
                    "Kaynak_Tipi": row.get("Kaynak_Tipi", "baslangic"),
                    "is_LGS": _safe_int(row.get("is_LGS", ""), default=0),
                    "egitim_agirligi": _safe_float(row.get("egitim_agirligi", ""), default=0.5)
                }
                questions.append(question)

        logger.info(f"Loaded {len(questions)} questions")

        self._questions_cache = questions
        return questions

    def get_cached_configs(self) -> dict[str, Any] | None:
        """Get cached configs if available"""
        return self._configs_cache

    def get_cached_questions(self) -> list[dict] | None:
        """Get cached questions if available"""
        return self._questions_cache

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._configs_cache = None
        self._questions_cache = None
        logger.debug("Config cache cleared")


# Convenience functions
def load_configs_from_env() -> dict[str, Any]:
    """
    Load configs from path specified in environment variable.

    Uses CONFIGS_PATH env var, defaults to 'data/configs.json'
    """
    import os
    path = os.getenv("CONFIGS_PATH", "data/configs.json")
    loader = ConfigLoader()
    return loader.load_configs(path)


def load_questions_from_env() -> list[dict[str, Any]]:
    """
    Load questions from path specified in environment variable.

    Uses QUESTIONS_CSV_PATH env var, defaults to 'data/dataset_ocr_li.csv'
    """
    import os
    path = os.getenv("QUESTIONS_CSV_PATH", "data/dataset_ocr_li.csv")
    loader = ConfigLoader()
    return loader.load_questions(path)

