"""Replace generated blocks in the README between <!-- NAME:START --> / <!-- NAME:END -->."""

from __future__ import annotations

import re


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {start} / {end} markers")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text, count=1)
