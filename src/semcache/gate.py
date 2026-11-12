"""Eval gate: fail the build when a measured number regresses.

Rules live in eval_gate.json, grouped by mode ("smoke" for CI, "real" after a real run):

    {"file": "reports/smoke/results.json", "path": "summary.all.attacks", "op": "<=",
     "value": 0, "why": "all four guardrails must stop every attack"}

`path` is dot-separated; a segment like `configs[name=baseline]` picks the list item whose
`name` is `baseline`. `value` may be a number or "$other.path" to compare two results.
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
}
_SELECT = re.compile(r"^(\w[\w ]*)\[(\w+)=([^\]]+)\]$")


def resolve(data: Any, path: str) -> Any:
    node = data
    for segment in path.split("."):
        m = _SELECT.match(segment)
        if m:
            key, field, wanted = m.groups()
            node = next((x for x in node[key] if str(x.get(field)) == wanted), None)
            if node is None:
                raise KeyError(f"no item with {field}={wanted} in {key}")
        else:
            node = node[segment]
    return node


@dataclass(frozen=True, slots=True)
class Outcome:
    rule: Mapping[str, Any]
    actual: Any
    expected: Any
    passed: bool
    error: str = ""


def evaluate(rules: Sequence[Mapping[str, Any]], root: Path) -> list[Outcome]:
    cache: dict[str, Any] = {}
    outcomes = []
    for rule in rules:
        try:
            if rule["file"] not in cache:
                cache[rule["file"]] = json.loads((root / rule["file"]).read_text(encoding="utf-8"))
            data = cache[rule["file"]]
            actual = resolve(data, rule["path"])
            value = rule["value"]
            expected = (
                resolve(data, value[1:])
                if isinstance(value, str) and value.startswith("$")
                else value
            )
            outcomes.append(Outcome(rule, actual, expected, OPS[rule["op"]](actual, expected)))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            outcomes.append(
                Outcome(rule, None, rule.get("value"), False, f"{type(exc).__name__}: {exc}")
            )
    return outcomes


def load_rules(config: Path, mode: str) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = json.loads(config.read_text(encoding="utf-8"))[mode]
    return rules
