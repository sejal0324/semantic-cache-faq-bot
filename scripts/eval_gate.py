"""Fail (exit 1) if measured results regress past the thresholds in eval_gate.json.

python scripts/eval_gate.py --mode smoke   # CI: deterministic smoke outputs
python scripts/eval_gate.py --mode real    # after `make experiments` on real models
"""

from __future__ import annotations

import argparse
import sys

from semcache.config import REPO_ROOT
from semcache.gate import evaluate, load_rules


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "real"], default="smoke")
    args = p.parse_args()
    outcomes = evaluate(load_rules(REPO_ROOT / "eval_gate.json", args.mode), REPO_ROOT)
    for o in outcomes:
        mark = "PASS" if o.passed else "FAIL"
        detail = o.error or f"{o.actual} {o.rule['op']} {o.expected}"
        print(f"[{mark}] {o.rule['path']}: {detail}  ({o.rule['why']})")
    failed = sum(not o.passed for o in outcomes)
    print(f"\n{len(outcomes) - failed}/{len(outcomes)} checks passed ({args.mode}).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
