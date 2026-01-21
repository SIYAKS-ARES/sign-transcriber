"""
LGS RAG Configuration
Pydantic Settings ile ortam degiskenleri yonetimi
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path


class Settings(BaseSettings):
    """Uygulama konfigurasyonlari"""

    # LLM Settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, env="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")
    llm_provider: Literal["openai", "anthropic", "gemini"] = Field("openai", env="LLM_PROVIDER")
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4000, env="LLM_MAX_TOKENS")

    # Embedding Settings
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_provider: Literal["openai", "huggingface"] = Field(
        "openai", env="EMBEDDING_PROVIDER"
    )

    # Vector Store Settings
    vector_store_type: Literal["faiss", "chroma"] = Field(
        "faiss", env="VECTOR_STORE_TYPE"
    )
    vector_store_path: Path = Field(Path("./vectorstore"), env="VECTOR_STORE_PATH")

    # Data Paths
    configs_path: Path = Field(
        Path("../lgs-model/outputs/configs.json"), env="CONFIGS_PATH"
    )
    questions_csv_path: Path = Field(
        Path("../lgs-model/data/processed/dataset_ocr_li.csv"),
        env="QUESTIONS_CSV_PATH",
    )
    generated_questions_path: Path = Field(
        Path("./data/generated_questions"), env="GENERATED_QUESTIONS_PATH"
    )

    # RAG Settings
    retrieval_top_k: int = Field(5, env="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    max_generation_attempts: int = Field(3, env="MAX_GENERATION_ATTEMPTS")

    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    debug: bool = Field(False, env="DEBUG")

    # Rate Limiting (opsiyonel)
    rate_limit_rpm: int | None = Field(None, env="RATE_LIMIT_RPM")
    rate_limit_tpm: int | None = Field(None, env="RATE_LIMIT_TPM")

    # Logging Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", env="LOG_LEVEL"
    )
    log_format: Literal["json", "text"] = Field("json", env="LOG_FORMAT")

    # Manim Settings
    manim_output_dir: Path = Field(
        Path("./generated_visuals"), env="MANIM_OUTPUT_DIR"
    )
    manim_quality: Literal["l", "m", "h"] = Field("l", env="MANIM_QUALITY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

