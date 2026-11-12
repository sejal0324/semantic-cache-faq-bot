"""All configuration comes from environment variables / `.env`, validated at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY_S = 86_400.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["ollama", "gemini", "fake"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"
    gemini_thinking_budget: int | None = 0
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_attempts: int = Field(default=4, ge=1, le=10)

    # Embeddings
    embedder: Literal["st", "fake"] = "st"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Cache
    cache_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    vector_store: Literal["faiss", "memory"] = "faiss"
    max_entries_per_tenant: int = Field(default=10_000, ge=1)
    ttl_policy_s: float = Field(default=7 * DAY_S, gt=0)
    ttl_pricing_s: float = Field(default=1 * DAY_S, gt=0)
    ttl_general_s: float = Field(default=30 * DAY_S, gt=0)
    writeback: bool = True

    # Service
    max_question_chars: int = Field(default=500, ge=1, le=10_000)
    log_level: str = "INFO"
    data_dir: Path = REPO_ROOT / "data"

    @model_validator(mode="after")
    def _check_provider(self) -> Settings:
        if self.llm_provider == "gemini" and not self.gemini_api_key.get_secret_value():
            raise ValueError("LLM_PROVIDER=gemini needs GEMINI_API_KEY (see .env.example)")
        return self

    def ttl_by_category(self) -> dict[str, float]:
        return {
            "policy": self.ttl_policy_s,
            "pricing": self.ttl_pricing_s,
            "general": self.ttl_general_s,
        }
