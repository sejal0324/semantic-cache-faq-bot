"""Wiring: build the cache, LLM and bot from Settings. Keeps construction out of the API."""

from __future__ import annotations

from semcache.bot import FAQBot, seed_faqs
from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.data import load_faqs
from semcache.embeddings import Embedder, build_embedder
from semcache.llm import LLMClient, build_llm


def build_cache(settings: Settings, embedder: Embedder) -> SemanticCache:
    return SemanticCache(
        embedder,
        threshold=settings.cache_threshold,
        store_kind=settings.vector_store,
        ttl_s_by_category=settings.ttl_by_category(),
        default_ttl_s=settings.ttl_generated_s,
        max_entries_per_tenant=settings.max_entries_per_tenant,
        max_tenants=settings.max_tenants,
        allowed_link_domains=settings.allowed_link_domains,
    )


def build_bot(
    settings: Settings, *, embedder: Embedder | None = None, llm: LLMClient | None = None
) -> FAQBot:
    cache = build_cache(settings, embedder or build_embedder(settings))
    seed_faqs(cache, settings.seed_tenant, load_faqs(settings.data_dir / "faqs.jsonl"))
    return FAQBot(
        cache,
        llm or build_llm(settings),
        writeback=settings.writeback,
        exact_fast_path=settings.exact_fast_path,
        single_flight=settings.single_flight,
    )
