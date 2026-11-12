"""Link extraction and domain allowlisting for answers that will be cached or shown."""

from __future__ import annotations

import re
from collections.abc import Iterable

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.IGNORECASE)
_HOST = re.compile(r"^(?:https?://)?([^/:?#]+)", re.IGNORECASE)


def hosts_in(text: str) -> list[str]:
    hosts: list[str] = []
    for url in _URL.findall(text):
        m = _HOST.match(url)
        if m:
            hosts.append(m.group(1).lower().rstrip("."))
    return hosts


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in allowed)


def has_disallowed_link(text: str, allowed: Iterable[str]) -> bool:
    allowed = [d.lower() for d in allowed]
    return any(not host_allowed(h, allowed) for h in hosts_in(text))
