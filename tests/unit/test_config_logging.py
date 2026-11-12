import json
import logging

import pytest
from pydantic import ValidationError

from semcache.config import Settings
from semcache.llm import FakeLLM, GeminiClient, OllamaClient, build_llm
from semcache.logging_utils import JsonFormatter, configure_logging, log_event, short_hash


def test_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CACHE_THRESHOLD", "0.9")
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake"
    assert s.cache_threshold == 0.9


def test_settings_rejects_bad_threshold() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cache_threshold=1.5)


def test_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="gpt-banana")


def test_gemini_without_key_fails_fast() -> None:
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, llm_provider="gemini", gemini_api_key="")


def test_secret_not_in_repr() -> None:
    s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="super-secret-key")
    assert "super-secret-key" not in repr(s)


def test_build_llm_picks_backend() -> None:
    assert isinstance(build_llm(Settings(_env_file=None, llm_provider="fake")), FakeLLM)
    assert isinstance(build_llm(Settings(_env_file=None, llm_provider="ollama")), OllamaClient)
    g = build_llm(Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k"))
    assert isinstance(g, GeminiClient)


def test_json_formatter_includes_fields() -> None:
    rec = logging.LogRecord("semcache", logging.INFO, __file__, 1, "ask", None, None)
    rec.fields = {"hit": True, "latency_ms": 1.5}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "ask"
    assert out["hit"] is True
    assert out["level"] == "INFO"


def test_log_event_emits_json(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging("INFO")
    logger = logging.getLogger("semcache.test")
    with caplog.at_level(logging.INFO, logger="semcache.test"):
        log_event(logger, "request", outcome="hit")
    assert caplog.records[-1].fields == {"outcome": "hit"}  # type: ignore[attr-defined]


def test_short_hash_stable_and_short() -> None:
    assert short_hash("abc") == short_hash("abc")
    assert len(short_hash("abc")) == 16
    assert short_hash("abc") != short_hash("abd")
