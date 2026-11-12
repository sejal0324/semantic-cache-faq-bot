"""Load simulation: real HTTP server (uvicorn) + fake backends with realistic latency.

    python loadtest/run_load.py --label baseline
    python loadtest/run_load.py --label tuned --env SINGLE_FLIGHT=true --env MAX_INFLIGHT_LLM=256

Each stage starts a fresh server (empty write-back cache), ramps up N simulated users, and
runs for --duration seconds. Users pick questions from a Zipf distribution (a few questions
are very popular, like real support traffic) and wait 0.5-2 s between questions.
The load generator runs on the same machine as the server, so results are a lower bound.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "loadtest"
STAGES = [50, 500, 1000, 2000, 5000]
BASE_ENV = {
    "LLM_PROVIDER": "fake",
    "EMBEDDER": "fake",
    "FAKE_LLM_LATENCY_S": "[0.3, 1.5]",  # typical hosted / local LLM response time
    "FAKE_EMBED_LATENCY_S": "0.01",  # ~10 ms CPU per embedding (MiniLM on CPU)
    "LOG_LEVEL": "WARNING",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def question_pool() -> list[str]:
    rows = [json.loads(line) for line in (REPO / "data" / "test_set.jsonl").open()]
    pool = [r["text"] for r in rows]
    pool += [f"Do you deliver to pincode {400000 + i}?" for i in range(100)]
    return pool


def start_server(port: int, extra_env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    env = {**os.environ, **BASE_ENV, **extra_env}
    log = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "semcache.api:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--backlog",
            "4096",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log,
        cwd=REPO,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start; see " + str(log_path))


async def user_loop(
    client: httpx.AsyncClient,
    url: str,
    pool: list[str],
    weights: list[float],
    rng: random.Random,
    start_delay: float,
    stop_at: float,
    results: list[tuple[float, str, Any]],
) -> None:
    await asyncio.sleep(start_delay)
    while time.perf_counter() < stop_at:
        q = rng.choices(pool, weights)[0]
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json={"tenant_id": "bazaarnow", "question": q})
            status = str(r.status_code)
            cached = r.json().get("cached") if r.status_code == 200 else None
        except httpx.HTTPError as exc:
            status, cached = type(exc).__name__, None
        results.append(((time.perf_counter() - t0) * 1000, status, cached))
        await asyncio.sleep(rng.uniform(0.5, 2.0))


async def sample_server(
    proc: psutil.Process, stop: asyncio.Event, out: list[tuple[float, float, float]]
) -> None:
    me = psutil.Process()  # the load generator itself
    proc.cpu_percent(None)
    me.cpu_percent(None)
    while not stop.is_set():
        await asyncio.sleep(0.5)
        try:
            out.append((proc.cpu_percent(None), proc.memory_info().rss / 1e6, me.cpu_percent(None)))
        except psutil.Error:
            return


async def run_stage(
    users: int, duration: float, ramp: float, extra_env: dict[str, str]
) -> dict[str, Any]:
    port = free_port()
    OUT.mkdir(parents=True, exist_ok=True)
    server = start_server(port, extra_env, OUT / "server.log")
    try:
        pool = question_pool()
        weights = [1 / (rank**1.1) for rank in range(1, len(pool) + 1)]
        rng = random.Random(42)
        rng.shuffle(pool)
        results: list[tuple[float, str, Any]] = []
        samples: list[tuple[float, float, float]] = []
        stop = asyncio.Event()
        limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)
        timeout = httpx.Timeout(30.0, connect=10.0, pool=30.0)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            sampler = asyncio.create_task(sample_server(psutil.Process(server.pid), stop, samples))
            t_start = time.perf_counter()
            stop_at = t_start + ramp + duration
            await asyncio.gather(
                *(
                    user_loop(
                        client,
                        f"http://127.0.0.1:{port}/ask",
                        pool,
                        weights,
                        random.Random(i),
                        ramp * i / users,
                        stop_at,
                        results,
                    )
                    for i in range(users)
                )
            )
            elapsed = time.perf_counter() - t_start
            stop.set()
            await sampler
            metrics = (await client.get(f"http://127.0.0.1:{port}/metrics")).json()
    finally:
        server.terminate()
        server.wait(timeout=10)

    ok = np.array([ms for ms, st, _ in results if st == "200"]) if results else np.array([])
    errors: dict[str, int] = {}
    for _, st, _ in results:
        if st != "200":
            errors[st] = errors.get(st, 0) + 1
    hits = sum(1 for _, st, c in results if st == "200" and c)
    n_ok = len(ok)
    cpu = [c for c, _, _ in samples] or [0.0]
    rss = [m for _, m, _ in samples] or [0.0]
    client_cpu = [c for _, _, c in samples] or [0.0]
    pct = (lambda p: round(float(np.percentile(ok, p)), 1)) if n_ok else (lambda p: None)
    return {
        "users": users,
        "requests": len(results),
        "rps": round(n_ok / elapsed, 1),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "error_rate": round(1 - n_ok / len(results), 4) if results else None,
        "errors": errors,
        "client_hit_rate": round(hits / n_ok, 3) if n_ok else None,
        "llm_calls": metrics.get("llm_calls"),
        "server_cpu_avg_pct": round(float(np.mean(cpu)), 1),
        "server_cpu_max_pct": round(float(np.max(cpu)), 1),
        "server_rss_max_mb": round(float(np.max(rss)), 1),
        "client_cpu_avg_pct": round(float(np.mean(client_cpu)), 1),
        "elapsed_s": round(elapsed, 1),
    }


def to_markdown(label: str, rows: list[dict[str, Any]], env: dict[str, str]) -> str:
    lines = [
        f"### {label}",
        "",
        f"Server env: `{json.dumps({**BASE_ENV, **env})}`",
        "",
        "| Users | Requests | RPS | p50 ms | p95 ms | p99 ms | Error rate | Errors | "
        "Hit rate | LLM calls | Server CPU avg/max % | Server RSS MB | Load-gen CPU % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['users']} | {r['requests']} | {r['rps']} | {r['p50_ms']} | {r['p95_ms']} "
            f"| {r['p99_ms']} | {r['error_rate']:.1%} | {r['errors'] or '-'} "
            f"| {r['client_hit_rate']} | {r['llm_calls']} "
            f"| {r['server_cpu_avg_pct']}/{r['server_cpu_max_pct']} | {r['server_rss_max_mb']} "
            f"| {r.get('client_cpu_avg_pct', '-')} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="baseline")
    p.add_argument("--stages", type=int, nargs="+", default=STAGES)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--ramp", type=float, default=5.0)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server")
    args = p.parse_args()
    env = dict(kv.split("=", 1) for kv in args.env)
    rows = []
    for users in args.stages:
        row = asyncio.run(run_stage(users, args.duration, args.ramp, env))
        print(json.dumps(row), flush=True)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    md = to_markdown(args.label, rows, env)
    (OUT / f"{args.label}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
