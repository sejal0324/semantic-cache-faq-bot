"""Negation guard.

Embedding models put "refund milega" and "refund nahi milega" very close together, because
almost every token is shared. Returning the cached answer for the negated question is a
false hit. The guard is deliberately simple: if exactly one of the two questions contains a
negation word, the cache is skipped and the question goes to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from semcache.text import tokenize

ENGLISH_NEGATIONS = frozenset(
    {"not", "no", "never", "without", "cannot", "dont", "cant", "wont", "isnt", "didnt", "doesnt"}
)
HINGLISH_NEGATIONS = frozenset({"nahi", "nhi", "nahin", "nai", "mat", "na"})


@dataclass(frozen=True, slots=True)
class NegationGuard:
    words: frozenset[str]

    @classmethod
    def default(cls) -> NegationGuard:
        return cls(ENGLISH_NEGATIONS | HINGLISH_NEGATIONS)

    def is_negated(self, text: str) -> bool:
        return any(tok in self.words or tok.endswith("n't") for tok in tokenize(text))

    def allows(self, question: str, cached_question: str) -> bool:
        """True if serving `cached_question`'s answer for `question` is safe w.r.t. negation."""
        return self.is_negated(question) == self.is_negated(cached_question)
