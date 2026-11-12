"""Run the evaluation.

python scripts/run_experiments.py           # real models (needs .env), fills README
python scripts/run_experiments.py --smoke   # fake backends, writes reports/smoke/ only
"""

from __future__ import annotations

import argparse
import asyncio

from semcache.config import REPO_ROOT, Settings
from semcache.experiments import DEFAULT_MODELS, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fake backends, no network")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-false-hits", type=int, default=1)
    args = parser.parse_args()

    settings = Settings()
    if args.smoke:
        settings = settings.model_copy(update={"llm_provider": "fake", "embedder": "fake"})
    results = asyncio.run(
        run_experiments(
            settings,
            smoke=args.smoke,
            out_dir=REPO_ROOT / "reports",
            readme=REPO_ROOT / "README.md",
            models=args.models,
            max_false_hits=args.max_false_hits,
        )
    )
    where = "reports/smoke/RESULTS.md" if args.smoke else "README.md and reports/"
    print(results.headline)
    print(f"Written to {where}")


if __name__ == "__main__":
    main()
