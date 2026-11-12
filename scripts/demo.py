"""Demo: the same question asked three ways. The first is a miss (goes to the LLM), the next
two are served from the cache in milliseconds.

    python scripts/demo.py            # uses .env (real embeddings + LLM)
    python scripts/demo.py --fake     # offline, fake backends
"""

from __future__ import annotations

import argparse
import asyncio
import json

from semcache.bot import FAQBot
from semcache.config import REPO_ROOT, Settings
from semcache.embeddings import build_embedder
from semcache.factory import build_cache
from semcache.llm import FakeLLM, build_llm

QUESTIONS = [
    "When will I get my refund?",
    "when will i get my refund??",
    "refund kab milega?",
]


def chosen_threshold(default: float) -> float:
    results = REPO_ROOT / "reports" / "results.json"
    if results.exists():
        return float(json.loads(results.read_text(encoding="utf-8"))["chosen"]["threshold"])
    return default


async def main(fake: bool) -> None:
    settings = Settings()
    if fake:
        settings = settings.model_copy(update={"llm_provider": "fake", "embedder": "fake"})
    threshold = chosen_threshold(settings.cache_threshold)
    cache = build_cache(
        settings.model_copy(update={"cache_threshold": threshold}), build_embedder(settings)
    )
    llm = FakeLLM(latency_s=(0.8, 0.8)) if fake else build_llm(settings)
    bot = FAQBot(cache, llm)  # empty cache on purpose: the first question must miss
    print(f"threshold={threshold:.2f}  embeddings={cache.embedder.name}  llm={llm.model}\n")
    for q in QUESTIONS:
        r = await bot.ask("demo", q)
        source = "CACHE" if r.cached else "LLM  "
        print(f"[{source}] {r.latency_ms:8.1f} ms  sim={r.similarity:.3f}  Q: {q}")
        print(f"         A: {r.answer[:110]}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fake", action="store_true")
    asyncio.run(main(p.parse_args().fake))
