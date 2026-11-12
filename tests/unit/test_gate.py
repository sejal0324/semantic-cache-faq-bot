import json
from pathlib import Path

import pytest

from semcache.config import REPO_ROOT
from semcache.gate import evaluate, load_rules, resolve

DATA = {
    "summary": {"all": {"attacks": 0}, "none": {"attacks": 10}},
    "configs": [{"name": "baseline", "agreement": 10}, {"name": "both", "agreement": 22}],
    "v2 heuristic": {"recall": 0.95},
}


def test_resolve_paths_and_selectors() -> None:
    assert resolve(DATA, "summary.none.attacks") == 10
    assert resolve(DATA, "configs[name=both].agreement") == 22
    assert resolve(DATA, "v2 heuristic.recall") == 0.95
    with pytest.raises(KeyError):
        resolve(DATA, "configs[name=nope].agreement")


def test_evaluate_numbers_references_and_errors(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text(json.dumps(DATA))
    rules = [
        {"file": "r.json", "path": "summary.all.attacks", "op": "<=", "value": 0},
        {
            "file": "r.json",
            "path": "configs[name=both].agreement",
            "op": ">=",
            "value": "$configs[name=baseline].agreement",
        },
        {"file": "r.json", "path": "summary.none.attacks", "op": "<", "value": 5},
        {"file": "missing.json", "path": "x", "op": "==", "value": 1},
    ]
    out = evaluate(rules, tmp_path)
    assert [o.passed for o in out] == [True, True, False, False]
    assert out[1].expected == 10
    assert "FileNotFoundError" in out[3].error


def test_repo_gate_config_is_valid() -> None:
    for mode in ("smoke", "real"):
        rules = load_rules(REPO_ROOT / "eval_gate.json", mode)
        assert rules and all({"file", "path", "op", "value", "why"} <= set(r) for r in rules)


def test_committed_smoke_results_pass_the_smoke_gate() -> None:
    failures = [
        o
        for o in evaluate(load_rules(REPO_ROOT / "eval_gate.json", "smoke"), REPO_ROOT)
        if not o.passed
    ]
    assert failures == []
