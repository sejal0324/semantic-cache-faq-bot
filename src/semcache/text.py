"""Text normalisation shared by the cache, guards and PII detector.

Everything that compares or stores user text goes through `normalize` first, so invisible
characters and look-alike punctuation can't be used to dodge the guards.
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e"), None)
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})
_TOKEN = re.compile(r"[a-z0-9']+")

# Look-alike letters (Cyrillic, Greek, Armenian) folded to Latin for *matching only*.
# Stored and displayed text is never folded, so real Russian or Greek text is not altered.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0443": "y",
        "\u0445": "x",
        "\u0456": "i",
        "\u0458": "j",
        "\u0455": "s",
        "\u04bb": "h",
        "\u0501": "d",
        "\u0578": "n",
        "\u057d": "u",
        "\u0585": "o",
        "\u03bf": "o",
        "\u03b1": "a",
        "\u03bd": "v",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u0410": "A",
        "\u0412": "B",
        "\u0415": "E",
        "\u041a": "K",
        "\u041c": "M",
        "\u041d": "H",
        "\u041e": "O",
        "\u0420": "P",
        "\u0421": "C",
        "\u0422": "T",
        "\u0425": "X",
        "\u039f": "O",
        "\u0391": "A",
    }
)


def normalize(text: str) -> str:
    """Remove invisible chars, NFKC-normalise, unify apostrophes, collapse whitespace."""
    t = text.translate(_INVISIBLE)
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(_QUOTES)
    return " ".join(t.split())


def skeleton(text: str) -> str:
    """Normalised text with look-alike letters folded to Latin. Use for matching rules."""
    return normalize(text).translate(_CONFUSABLES)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(skeleton(text).lower())
