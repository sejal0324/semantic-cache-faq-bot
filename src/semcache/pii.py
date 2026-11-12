"""Detects user-specific data that must never be written to a shared cache.

Limits (documented on purpose): this is pattern matching, not NER. It catches the formats
this shop uses (order IDs, Indian mobile numbers, emails, "my name is ..." phrasing). It will
miss a bare name ("Priya here, refund?") or an address. That is why the cache only stores
answers generated for questions that pass this check, and the README calls this out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from semcache.text import skeleton

_DIGIT_SEPARATORS = re.compile(r"(?<=\d)[\s.\-]+(?=\d)")

DEFAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    # Shop order IDs look like OD + 10 or more digits.
    "order_id": re.compile(r"\bOD[\s-]?\d{10,}\b", re.IGNORECASE),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # Indian mobile: optional +91 / 0 prefix, starts with 6-9, 10 digits, optional 5+5 split.
    "phone": re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "name": re.compile(r"\b(?:my name is|i am called|mera naam|mera nam)\s+\w+", re.IGNORECASE),
}


@dataclass(frozen=True, slots=True)
class PIIDetector:
    patterns: dict[str, re.Pattern[str]] = field(default_factory=lambda: dict(DEFAULT_PATTERNS))

    def find(self, text: str) -> list[str]:
        t = skeleton(text)
        # "98765 43210", "9 8 7 6 5 ...", "98765.43210" -> "9876543210" for a second pass.
        collapsed = _DIGIT_SEPARATORS.sub("", t)
        return [
            kind
            for kind, pattern in self.patterns.items()
            if pattern.search(t) or pattern.search(collapsed)
        ]

    def contains_pii(self, text: str) -> bool:
        return bool(self.find(text))
