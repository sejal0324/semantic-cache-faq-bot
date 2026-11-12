"""LLM client abstraction: one async interface, three backends (Ollama, Gemini, Fake).

Design notes
- Async-first: the bot is I/O bound on the LLM, so one event loop can hold many waiting requests.
- Retries only on transient failures (429, 5xx, timeouts, connection errors) with capped
  exponential backoff and full jitter, so many clients don't retry in lockstep.
- Error messages never include the response body: providers can echo prompts or keys back.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

import httpx

if TYPE_CHECKING:
    from semcache.config import Settings

Role = Literal["system", "user", "assistant"]
T = TypeVar("T")
SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LLMError(RuntimeError):
    """Non-retryable failure: bad request, auth, malformed response, or retries exhausted."""


class RetryableLLMError(LLMError):
    """Transient failure: rate limit, 5xx, timeout, connection error."""


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough token count for backends that don't report usage (fake LLM)."""
    return len(text.split())


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        cap = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return rng.uniform(0.0, cap)


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: SleepFn = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography
    last: Exception | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except RetryableLLMError as exc:
            last = exc
            if attempt < policy.max_attempts - 1:
                await sleep(policy.delay_for(attempt, rng))
    raise LLMError(f"LLM call failed after {policy.max_attempts} attempts") from last


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableLLMError(f"transient HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise LLMError(f"HTTP {resp.status_code}")


class _HTTPBackend:
    """Shared plumbing for HTTP backends: timeout, retries, latency, error mapping."""

    def __init__(
        self,
        *,
        model: str,
        timeout_s: float,
        retry: RetryPolicy,
        transport: httpx.AsyncBaseTransport | None,
        sleep: SleepFn,
    ) -> None:
        self._model = model
        self._retry = retry
        self._sleep = sleep
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {}

    def _payload(
        self, messages: Sequence[Message], temperature: float, max_tokens: int
    ) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _parse(self, data: Any) -> tuple[str, int, int]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> LLMResponse:
        async def once() -> LLMResponse:
            start = time.perf_counter()
            try:
                resp = await self._client.post(
                    self._url(),
                    json=self._payload(messages, temperature, max_tokens),
                    headers=self._headers(),
                )
            except httpx.TransportError as exc:
                raise RetryableLLMError(type(exc).__name__) from exc
            _raise_for_status(resp)
            try:
                text, tokens_in, tokens_out = self._parse(resp.json())
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise LLMError("malformed LLM response") from exc
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(text, tokens_in, tokens_out, latency_ms, self._model)

        return await with_retries(once, self._retry, sleep=self._sleep)


class OllamaClient(_HTTPBackend):
    """Local models via Ollama's /api/chat endpoint (free, offline)."""

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout_s: float = 120.0,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        super().__init__(
            model=model,
            timeout_s=timeout_s,
            retry=retry or RetryPolicy(),
            transport=transport,
            sleep=sleep,
        )
        self._host = host.rstrip("/")

    def _url(self) -> str:
        return f"{self._host}/api/chat"

    def _payload(
        self, messages: Sequence[Message], temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def _parse(self, data: Any) -> tuple[str, int, int]:
        text = data["message"]["content"]
        if not isinstance(text, str):
            raise TypeError("content is not a string")
        return text, int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))


class GeminiClient(_HTTPBackend):
    """Google Gemini via the REST generateContent endpoint. Key goes in a header, not the URL."""

    BASE_URL = "https://generativelanguage.googleapis.com"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int | None = 0,
        timeout_s: float = 60.0,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini backend")
        super().__init__(
            model=model,
            timeout_s=timeout_s,
            retry=retry or RetryPolicy(),
            transport=transport,
            sleep=sleep,
        )
        self._api_key = api_key
        self._thinking_budget = thinking_budget

    def _url(self) -> str:
        return f"{self.BASE_URL}/v1beta/models/{self._model}:generateContent"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def _payload(
        self, messages: Sequence[Message], temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        system = [m.content for m in messages if m.role == "system"]
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self._thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}
        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return payload

    def _parse(self, data: Any) -> tuple[str, int, int]:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return (
            text,
            int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)),
        )


Responder = Callable[[Sequence[Message]], str]


def _echo(messages: Sequence[Message]) -> str:
    last = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return f"[fake answer] {last[:200]}"


class FakeLLM:
    """Deterministic in-process LLM for tests, CI, smoke runs and load tests.

    `latency_s` simulates model latency, `failure_rate` simulates 429s (which go through the
    same retry path as real backends).
    """

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        model: str = "fake-llm",
        latency_s: tuple[float, float] = (0.0, 0.0),
        failure_rate: float = 0.0,
        seed: int = 0,
        retry: RetryPolicy | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._responder = responder or _echo
        self._model = model
        self._latency = latency_s
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)  # noqa: S311 - simulation only
        self._retry = retry or RetryPolicy(max_attempts=4, base_delay_s=0.05, max_delay_s=0.5)
        self._sleep = sleep
        self.calls = 0
        self.failures = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def latency_s(self) -> tuple[float, float]:
        return self._latency

    async def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> LLMResponse:
        async def once() -> LLMResponse:
            self.calls += 1
            start = time.perf_counter()
            if self._failure_rate and self._rng.random() < self._failure_rate:
                self.failures += 1
                raise RetryableLLMError("simulated 429")
            delay = self._rng.uniform(*self._latency)
            if delay > 0:
                await self._sleep(delay)
            text = self._responder(messages)
            prompt = " ".join(m.content for m in messages)
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text, estimate_tokens(prompt), estimate_tokens(text), latency_ms, self._model
            )

        return await with_retries(once, self._retry, sleep=self._sleep, rng=self._rng)


def build_llm(settings: Settings) -> LLMClient:
    retry = RetryPolicy(max_attempts=settings.llm_max_attempts)
    if settings.llm_provider == "ollama":
        return OllamaClient(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout_s=settings.llm_timeout_s,
            retry=retry,
        )
    if settings.llm_provider == "gemini":
        return GeminiClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            thinking_budget=settings.gemini_thinking_budget,
            timeout_s=settings.llm_timeout_s,
            retry=retry,
        )
    return FakeLLM(latency_s=settings.fake_llm_latency_s)
