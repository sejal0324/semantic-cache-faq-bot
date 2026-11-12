"""Typed, validated loaders for the versioned JSONL datasets in data/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

M = TypeVar("M", bound=BaseModel)
MAX_DATASET_BYTES = 5 * 1024 * 1024


class FAQ(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^f\d{2,}$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: Literal["policy", "pricing", "general"]
    topic: str = ""


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^t\d{3}$")
    text: str = Field(min_length=1)
    expected_faq_id: str  # an FAQ id, or "none" if the cache must NOT answer it
    lang: Literal["en", "hinglish"]
    kind: Literal["paraphrase", "negation", "near_miss", "out_of_scope"]

    @property
    def should_hit(self) -> bool:
        return self.expected_faq_id != "none"


def load_jsonl(path: Path, model: type[M], *, max_bytes: int = MAX_DATASET_BYTES) -> list[M]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path.name} is too large ({path.stat().st_size} bytes)")
    rows: list[M] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid row: {exc}") from exc
    return rows


def load_faqs(path: Path) -> list[FAQ]:
    return load_jsonl(path, FAQ)


def load_test_set(path: Path) -> list[EvalQuestion]:
    return load_jsonl(path, EvalQuestion)
