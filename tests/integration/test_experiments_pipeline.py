"""The whole experiment pipeline on fake backends, against copies of the real datasets."""

import json
import shutil
from pathlib import Path

import pytest

from semcache.config import REPO_ROOT, Settings
from semcache.embeddings import FakeEmbedder
from semcache.experiments import run_experiments
from semcache.llm import FakeLLM

README = """# x
<!-- HEADLINE:START -->
Not yet run with real models.
<!-- HEADLINE:END -->
<!-- RESULTS:START -->
Not yet run with real models.
<!-- RESULTS:END -->
"""


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    return tmp_path


def settings(workdir: Path, provider: str = "fake") -> Settings:
    return Settings(
        _env_file=None, llm_provider=provider, embedder="fake", data_dir=workdir / "data"
    )  # type: ignore[arg-type]


async def test_smoke_run_writes_only_to_smoke_dir(workdir: Path) -> None:
    results = await run_experiments(
        settings(workdir), smoke=True, out_dir=workdir / "reports", readme=workdir / "README.md"
    )
    smoke = workdir / "reports" / "smoke"
    assert (smoke / "results.json").exists()
    md = (smoke / "RESULTS.md").read_text(encoding="utf-8")
    assert "fake-backend smoke run, not a real measurement" in md
    assert (workdir / "README.md").read_text(encoding="utf-8") == README, "README untouched"
    assert not (workdir / "reports" / "results.json").exists()
    data = json.loads((smoke / "results.json").read_text(encoding="utf-8"))
    assert data["mode"] == "smoke"
    assert len(results.rows) == 2 * 5  # guard on/off x 5 thresholds
    assert {r.threshold for r in results.rows} == {0.75, 0.8, 0.85, 0.9, 0.95}


async def test_real_mode_fills_readme(workdir: Path) -> None:
    results = await run_experiments(
        settings(workdir, provider="ollama"),
        smoke=False,
        out_dir=workdir / "reports",
        readme=workdir / "README.md",
        models=["model-a", "model-b"],
        embedder_factory=lambda name: FakeEmbedder(dim=128 if name == "model-a" else 256),
        llm=FakeLLM(model="stub-llm"),
    )
    readme = (workdir / "README.md").read_text(encoding="utf-8")
    assert "Not yet run" not in readme
    assert results.headline in readme
    assert "| Threshold |" in readme
    assert (workdir / "reports" / "results.json").exists()
    assert (workdir / "reports" / "per_question.jsonl").exists()
    assert {r.model for r in results.rows} == {"model-a", "model-b"}
    assert results.cost_per_1k_before >= results.cost_per_1k_after > 0  # fake embedder: no hits
    assert "per 1,000 questions" in readme


async def test_real_mode_refuses_fake_llm(workdir: Path) -> None:
    with pytest.raises(SystemExit, match="LLM_PROVIDER"):
        await run_experiments(
            settings(workdir, provider="fake"),
            smoke=False,
            out_dir=workdir / "reports",
            readme=workdir / "README.md",
        )
