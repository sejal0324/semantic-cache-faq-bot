import json
import random

import httpx
import pytest

from semcache.llm import (
    FakeLLM,
    GeminiClient,
    LLMError,
    Message,
    OllamaClient,
    RetryableLLMError,
    RetryPolicy,
    estimate_tokens,
    with_retries,
)

MSGS = [Message("system", "be brief"), Message("user", "hello?")]


async def no_sleep(_: float) -> None:
    return None


def transport(handler):  # type: ignore[no-untyped-def]
    return httpx.MockTransport(handler)


# ---------- retry policy ----------


def test_retry_delay_is_capped_and_non_negative() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=3.0)
    rng = random.Random(0)
    for attempt in range(10):
        d = policy.delay_for(attempt, rng)
        assert 0.0 <= d <= 3.0


async def test_with_retries_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableLLMError("429")
        return "ok"

    out = await with_retries(flaky, RetryPolicy(max_attempts=4), sleep=no_sleep)
    assert out == "ok"
    assert calls["n"] == 3


async def test_with_retries_gives_up_after_max_attempts() -> None:
    calls = {"n": 0}

    async def always_fail():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise RetryableLLMError("503")

    with pytest.raises(LLMError, match="after 3 attempts"):
        await with_retries(always_fail, RetryPolicy(max_attempts=3), sleep=no_sleep)
    assert calls["n"] == 3


async def test_with_retries_does_not_retry_permanent_errors() -> None:
    calls = {"n": 0}

    async def bad_request():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise LLMError("HTTP 400")

    with pytest.raises(LLMError):
        await with_retries(bad_request, RetryPolicy(max_attempts=5), sleep=no_sleep)
    assert calls["n"] == 1


# ---------- Ollama ----------


async def test_ollama_builds_request_and_parses_response() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"message": {"content": "hi there"}, "prompt_eval_count": 7, "eval_count": 3},
        )

    client = OllamaClient(
        host="http://ollama:11434/", model="qwen2.5:3b", transport=transport(handler)
    )
    resp = await client.complete(MSGS, temperature=0.0, max_tokens=64)
    await client.aclose()
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["model"] == "qwen2.5:3b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"] == {"temperature": 0.0, "num_predict": 64}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert resp.text == "hi there"
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
    assert resp.model == "qwen2.5:3b"
    assert resp.latency_ms >= 0


async def test_ollama_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = OllamaClient(host="http://x", model="m", transport=transport(handler), sleep=no_sleep)
    resp = await client.complete(MSGS)
    assert resp.text == "ok"
    assert calls["n"] == 2


async def test_ollama_400_is_not_retried_and_body_not_leaked() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="secret-prompt-echo")

    client = OllamaClient(host="http://x", model="m", transport=transport(handler))
    with pytest.raises(LLMError) as exc:
        await client.complete(MSGS)
    assert calls["n"] == 1
    assert "secret-prompt-echo" not in str(exc.value)


async def test_ollama_transport_error_is_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    client = OllamaClient(
        host="http://x",
        model="m",
        transport=transport(handler),
        retry=RetryPolicy(max_attempts=2),
        sleep=no_sleep,
    )
    with pytest.raises(LLMError):
        await client.complete(MSGS)
    assert calls["n"] == 2


async def test_malformed_response_raises_llm_error() -> None:
    client = OllamaClient(
        host="http://x",
        model="m",
        transport=transport(lambda r: httpx.Response(200, json={"nope": 1})),
    )
    with pytest.raises(LLMError, match="malformed"):
        await client.complete(MSGS)


# ---------- Gemini ----------


async def test_gemini_builds_request_and_parses_response() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "thinking...", "thought": True}, {"text": "A"}]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2},
            },
        )

    client = GeminiClient(api_key="k-123", model="gemini-x", transport=transport(handler))
    msgs = [*MSGS, Message("assistant", "earlier"), Message("user", "again")]
    resp = await client.complete(msgs, temperature=0.2, max_tokens=50)
    assert seen["url"].endswith("/v1beta/models/gemini-x:generateContent")
    assert "key=" not in seen["url"], "API key must not be in the URL (it ends up in logs)"
    assert seen["key"] == "k-123"
    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["generationConfig"]["maxOutputTokens"] == 50
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert resp.text == "A"
    assert (resp.input_tokens, resp.output_tokens) == (11, 2)


async def test_gemini_without_thinking_budget_omits_it() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "x"}]}}]})

    client = GeminiClient(
        api_key="k", model="m", thinking_budget=None, transport=transport(handler)
    )
    await client.complete([Message("user", "q")])
    assert "thinkingConfig" not in seen["body"]["generationConfig"]
    assert "systemInstruction" not in seen["body"]


async def test_gemini_blocked_response_raises() -> None:
    client = GeminiClient(
        api_key="k",
        model="m",
        transport=transport(lambda r: httpx.Response(200, json={"promptFeedback": {}})),
    )
    with pytest.raises(LLMError):
        await client.complete(MSGS)


def test_gemini_requires_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="", model="m")


# ---------- Fake ----------


async def test_fake_llm_is_deterministic_and_counts_calls() -> None:
    llm = FakeLLM(responder=lambda msgs: "answer:" + msgs[-1].content)
    r1 = await llm.complete(MSGS)
    r2 = await llm.complete(MSGS)
    assert r1.text == r2.text == "answer:hello?"
    assert llm.calls == 2
    assert r1.output_tokens == estimate_tokens(r1.text)


async def test_fake_llm_default_responder() -> None:
    llm = FakeLLM()
    r = await llm.complete([Message("user", "where is my parcel")])
    assert "where is my parcel" in r.text
    assert llm.model == "fake-llm"


async def test_fake_llm_simulated_failures_are_retried() -> None:
    llm = FakeLLM(failure_rate=0.5, seed=1, retry=RetryPolicy(max_attempts=10), sleep=no_sleep)
    for _ in range(20):
        await llm.complete(MSGS)
    assert llm.calls > 20  # some attempts failed and were retried
    assert llm.failures == llm.calls - 20


async def test_fake_llm_latency_uses_sleep() -> None:
    slept: list[float] = []

    async def record(s: float) -> None:
        slept.append(s)

    llm = FakeLLM(latency_s=(0.3, 0.3), sleep=record)
    await llm.complete(MSGS)
    assert slept == [0.3]


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a b c") == 3
