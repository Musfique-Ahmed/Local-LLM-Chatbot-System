"""Benchmark Redis vs MongoDB stores for chatbot history operations.

Workload per iteration: get -> append 2 turns -> save -> clear.
Runs N=200 iterations per backend by default. Writes results to
benchmarks/results.json in a shape ready for COMPARISON.md and dashboard.html.

Usage:
    python benchmarks/bench_stores.py --backend both --iters 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

# Ensure project root is on sys.path when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chatbot import config  # noqa: E402
from chatbot.store import make_store  # noqa: E402


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile; pct in [0, 100]."""
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize(samples_ms: list[float]) -> dict:
    """Return {p50, p95, p99, mean, stddev, ops_per_sec} from ms samples."""
    if not samples_ms:
        return {
            "p50": 0.0, "p95": 0.0, "p99": 0.0,
            "mean": 0.0, "stddev": 0.0, "ops_per_sec": 0.0,
        }
    mean = statistics.fmean(samples_ms)
    stddev = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    # ops/sec derived from the MEAN latency of a single op
    ops = (1000.0 / mean) if mean > 0 else 0.0
    return {
        "p50": round(_percentile(samples_ms, 50), 4),
        "p95": round(_percentile(samples_ms, 95), 4),
        "p99": round(_percentile(samples_ms, 99), 4),
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "ops_per_sec": round(ops, 1),
    }


def _bench_backend(backend: str, iters: int) -> dict:
    """Run the workload against one backend and return its metric block."""
    # Force the right backend via env (config is read at import time of modules).
    os.environ["STORE_BACKEND"] = backend
    # Re-import to pick up the new STORE_BACKEND (config reads env on import).
    import importlib
    from chatbot import store as store_mod

    importlib.reload(store_mod)

    if backend == "mongo":
        # Need a URI for Mongo. Reuse the same env var the server uses.
        if not config.MONGO_URI:
            print("  ERROR: MONGO_URI not set; skipping mongo benchmark.",
                  file=sys.stderr)
            return {}

    store = store_mod.make_store()
    samples: dict[str, list[float]] = {"get": [], "save": [], "clear": []}
    user = f"bench_user_{int(time.time())}"

    # Warm-up to avoid first-call connection setup skewing results.
    for _ in range(5):
        store.save_history(user, [{"role": "user", "content": "warm"}])
        store.get_history(user)
        store.clear_history(user)

    for i in range(iters):
        t = time.perf_counter()
        store.get_history(user)
        samples["get"].append((time.perf_counter() - t) * 1000.0)

        history = store.get_history(user)
        history.append({"role": "user", "content": f"msg {i}"})
        history.append({"role": "assistant", "content": f"reply {i}"})

        t = time.perf_counter()
        store.save_history(user, history)
        samples["save"].append((time.perf_counter() - t) * 1000.0)

        t = time.perf_counter()
        store.clear_history(user)
        samples["clear"].append((time.perf_counter() - t) * 1000.0)

    store.clear_history(user)  # cleanup

    block: dict = {}
    for op, ms_list in samples.items():
        block[op] = _summarize(ms_list)
    # A normal chat turn (without clear) is get + save = 2 calls.
    block["calls_per_chat_turn"] = 2
    return block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["redis", "mongo", "both"],
                        default="both")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default=os.path.join(ROOT, "benchmarks",
                                                       "results.json"))
    args = parser.parse_args()

    backends = ["redis", "mongo"] if args.backend == "both" else [args.backend]
    results: dict = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": args.iters,
        "model": config.MODEL_NAME,
        "redis": {"host": config.REDIS_HOST, "port": config.REDIS_PORT},
        "mongo": {
            "uri_set": bool(config.MONGO_URI),
            "db": config.MONGO_DB,
            "collection": config.MONGO_COLLECTION,
        },
    }}

    for backend in backends:
        print(f"benchmarking {backend} ...", flush=True)
        block = _bench_backend(backend, args.iters)
        if block:
            results[backend] = block
            for op, m in block.items():
                if isinstance(m, dict):
                    print(f"  {op:5s} p50={m['p50']:7.3f}ms "
                          f"p95={m['p95']:7.3f}ms "
                          f"p99={m['p99']:7.3f}ms "
                          f"ops/s={m['ops_per_sec']:8.0f}")
        else:
            print(f"  {backend}: skipped")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
