from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = "Multi-Document RAG QA System"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    generation_model: str = "google/flan-t5-base"
    generation_device: int = -1
    generation_max_new_tokens: int = Field(default=128, ge=1, le=512)
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    sample_documents_dir: Path = PROJECT_ROOT / "data" / "sample_documents"
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstore"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = Field(default=5, ge=1, le=5)
    min_relevance_score: float = 0.15
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
