"""Malformed and hostile HTTP input."""

import pytest
from fastapi.testclient import TestClient

from semcache.api import create_app
from semcache.config import Settings
from semcache.factory import build_bot
from semcache.llm import FakeLLM

SETTINGS = Settings(_env_file=None, llm_provider="fake", embedder="fake", max_question_chars=100)


@pytest.fixture
def client() -> TestClient:
    def crash(_: object) -> str:
        raise RuntimeError("Traceback: secret internals at /srv/app")

    bot = build_bot(SETTINGS, llm=FakeLLM(responder=crash))  # type: ignore[arg-type]
    return TestClient(create_app(bot=bot, settings=SETTINGS), raise_server_exceptions=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"tenant_id": "../../etc", "question": "hi"},
        {"tenant_id": "a" * 65, "question": "hi"},
        {"tenant_id": "", "question": "hi"},
        {"tenant_id": "ok", "question": ""},
        {"tenant_id": "ok", "question": "   \u200b  "},
        {"tenant_id": "ok", "question": "x" * 101},
        {"tenant_id": "ok", "question": "hi", "threshold": 0.0},
        {"tenant_id": "ok", "question": 123},
        {"tenant_id": "ok"},
    ],
)
def test_bad_input_rejected_with_422(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_nfkc_expansion_cannot_bypass_length_limit(client: TestClient) -> None:
    # U+FDFA is one character that NFKC expands to 18. 10 of them = 180 chars > 100.
    r = client.post("/ask", json={"tenant_id": "ok", "question": "\ufdfa" * 10})
    assert r.status_code == 422


def test_hard_cap_before_normalisation(client: TestClient) -> None:
    r = client.post("/ask", json={"tenant_id": "ok", "question": "a" * 5_000})
    assert r.status_code == 422


def test_unexpected_error_is_generic_500(client: TestClient) -> None:
    r = client.post("/ask", json={"tenant_id": "ok", "question": "a brand new question"})
    assert r.status_code == 500
    assert "secret internals" not in r.text
    assert "Traceback" not in r.text


def test_metrics_do_not_expose_questions(client: TestClient) -> None:
    client.post("/ask", json={"tenant_id": "bazaarnow", "question": "How do I track my order?"})
    assert "track" not in client.get("/metrics").text
