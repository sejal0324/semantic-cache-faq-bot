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

    # Fake backends (tests, smoke runs, load tests): simulated latency
    fake_llm_latency_s: tuple[float, float] = (0.0, 0.0)
    fake_embed_latency_s: float = Field(default=0.0, ge=0.0)

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
    ttl_generated_s: float = Field(default=1 * DAY_S, gt=0)
    max_tenants: int = Field(default=1_000, ge=1)
    allowed_link_domains: frozenset[str] = frozenset({"bazaarnow.in"})
    seed_tenant: str = "bazaarnow"
    writeback: bool = True
    # Scaling (see LOAD_TEST.md)
    exact_fast_path: bool = True  # exact repeats skip the embedding model
    single_flight: bool = True  # concurrent identical misses share one LLM call
    max_inflight_requests: int = Field(default=0, ge=0)  # 0 = no load shedding

    # Cost model (USD per 1M tokens) used only for the cost report. Set your provider's price.
    price_in_per_1m: float = Field(default=0.30, ge=0.0)
    price_out_per_1m: float = Field(default=2.50, ge=0.0)

    # Service
    admin_token: SecretStr = SecretStr("")  # enables DELETE /cache/{tenant}; empty = disabled
    max_question_chars: int = Field(default=500, ge=1, le=10_000)
    log_level: str = "INFO"
    tracing: Literal["off", "console", "otlp"] = "off"
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
            "generated": self.ttl_generated_s,
        }
