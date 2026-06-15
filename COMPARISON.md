# Redis vs MongoDB Atlas M0 — Store Backend Comparison

A side-by-side comparison of two conversation-history backends for the local LLM
chatbot: **Redis** (running on `localhost`) and **MongoDB Atlas M0 free tier**
(documented expected behavior — also bench-tested against a local MongoDB 7
container to confirm correctness).

---

## 1. Architecture

```
                         ┌─────────────────────┐
   TCP client            │  asyncio TCP server │             ┌──────────────┐
   (tcp_client) ───────▶ │  (chatbot/server.py)│ ── prompt ─▶│   Ollama     │
                         │                     │ ◀─ text ─── │ gemma3:4b    │
                         │  ┌───────────────┐  │             │ (100% GPU)   │
                         │  │ Store backend │  │             └──────────────┘
                         │  └───────┬───────┘  │
                         └──────────┼──────────┘
                                    │
                       STORE_BACKEND│= "redis" or "mongo"
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
       ┌─────────────────┐                     ┌──────────────────────┐
       │      Redis      │                     │  MongoDB Atlas M0    │
       │  in-mem KV store│                     │  document store     │
       │  key chat:{uid} │                     │  coll history        │
       │  value: JSON [] │                     │  doc {_id,messages}  │
       └─────────────────┘                     └──────────────────────┘
```

Both backends implement the same 3-method interface (`get_history`,
`save_history`, `clear_history`) defined in `chatbot/store.py`. The server
picks one at startup via `STORE_BACKEND`.

---

## 2. Measured latency (this machine, localhost → localhost)

200 iterations of `get → append 2 turns → save → clear`, per backend. Full raw
output: `benchmarks/results.json`.

| Op   | Redis p50 (ms) | Redis p95 (ms) | Redis p99 (ms) | Mongo p50 (ms) | Mongo p95 (ms) | Mongo p99 (ms) |
|------|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| get  | 0.78           | 0.94           | 1.07           | 0.81           | 1.02           | 1.10           |
| save | 0.80           | 0.99           | 1.09           | 0.83           | 1.04           | 1.12           |
| clear| 0.80           | 0.95           | 1.29           | 0.79           | 0.97           | 1.18           |

**On localhost → localhost the two are essentially tied** (~0.8 ms p50). The
real difference appears over a network. Atlas M0 from a same-region client
documents **p50 ≈ 15–40 ms, p99 ≈ 80–150 ms** (cross-region 2–5× worse). Redis
in production over a network typically adds 0.5–2 ms p50.

| Op   | Redis ops/sec | Mongo ops/sec |
|------|--------------:|--------------:|
| get  | 1,242         | 1,191         |
| save | 1,218         | 1,185         |
| clear| 1,235         | 1,222         |

---

## 3. Side-by-side table

| Dimension                | Redis                                          | MongoDB Atlas M0                                 |
|--------------------------|------------------------------------------------|--------------------------------------------------|
| Data model               | Key → string (JSON-encoded list)               | Document per user (embedded array of turns)      |
| Query language           | `GET / SET / DEL` (one key per user)           | `find_one / replace_one(upsert) / delete_one`    |
| Schema flexibility       | None (string is opaque)                        | Document-shaped, easy to evolve (add fields)     |
| Latency — same-host p50  | ~0.8 ms (this run)                             | ~0.8 ms (this run)                               |
| Latency — over WAN p50   | 0.5–2 ms                                       | 15–40 ms (Atlas M0, same region)                 |
| Persistence              | None by default; RDB/AOF opt-in                | Durable by default; 7-day rolling backup on M0   |
| Free-tier limits         | None (self-hosted)                             | 512 MB, shared RAM, ~100 conn, 100 w/1000 r ops/s|
| Ops per chat turn        | 2 (1 GET + 1 SET)                              | 2 (1 find_one + 1 replace_one)                   |
| Dependency size          | `redis` (≈ 200 KB)                             | `pymongo` + `dnspython` (≈ 1.5 MB)               |
| Setup effort             | `docker run -d -p 6379:6379 redis:7-alpine`     | Atlas UI → M0 cluster → user → IP allowlist → URI|
| Multi-user queryability  | Only via `KEYS chat:*` (don't in prod)         | Native: `{messages.0.content: "blue"}` etc.     |
| Failure mode             | Volatile by default; data loss on crash        | Replica set with automatic failover              |

---

## 4. When to pick which

**Pick Redis when:**
- History is **ephemeral** (a session, lost on logout is fine)
- You need **sub-millisecond** read latency
- You already run Redis for caching/sessions
- You're optimising for the **hottest path** in a high-RPS service

**Pick MongoDB Atlas M0 when:**
- You want **durable** history that survives crashes
- You might want to **query** history later (e.g. "all messages containing
  'invoice'", or aggregate per-user stats)
- You'd rather not run a **second service** locally (Atlas is fully managed;
  free tier is plenty for one user)
- You're building toward a **multi-user** product and need real persistence

**For this specific project** (single user, local LLM, gemma3:4b, ~3 s/turn
from Ollama anyway):
- The chatbot's chat-turn latency is **dominated by inference** (2–3 s), not
  the 0.8 ms store call. Neither backend is a bottleneck.
- The deciding factor is **ops overhead**: Redis requires you to run a local
  container; MongoDB Atlas is zero-install on a fresh machine (just set
  `MONGO_URI`).
- Recommendation: **MongoDB Atlas M0** wins on simplicity + durability. Redis
  wins on raw speed, but that speed is invisible at human typing cadence.

---

## 5. How to switch backends

```bash
# Redis (default; needs redis running on localhost:6379)
python -m chatbot.server

# MongoDB Atlas M0 (set the SRV URI from your Atlas dashboard)
export STORE_BACKEND=mongo
export MONGO_URI="mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
python -m chatbot.server
```

Both verified working with the same 4-turn test (context retention, clear,
reset). See `tests/test_chatbot.py` and `tests/test_chatbot_mongo.py` in the
project root's local temp scripts.

---

## 6. Reproducing the benchmark

```bash
# Both backends must be running locally:
docker run -d -p 6379:6379 --name redis-chatbot redis:7-alpine
docker run -d -p 27017:27017 --name mongo-chatbot mongo:7

# Run:
STORE_BACKEND=mongo MONGO_URI=mongodb://localhost:27017/ \
    python benchmarks/bench_stores.py --backend both --iters 200
# → writes benchmarks/results.json
```

To compare against **Atlas M0** instead of the local container, just point
`MONGO_URI` at your Atlas cluster. The numbers in `results.json` will jump
into the 10–50 ms range immediately.

---

## 7. Files added in this comparison

| Path                              | Purpose                                   |
|-----------------------------------|-------------------------------------------|
| `chatbot/store.py`                | Store Protocol + factory                  |
| `chatbot/mongo_store.py`          | MongoDB Atlas M0 implementation           |
| `chatbot/redis_store.py`          | Refactored: now also exposes `RedisStore` |
| `chatbot/config.py`               | Added `STORE_BACKEND`, `MONGO_URI`, `MONGO_DB`, `MONGO_COLLECTION` |
| `chatbot/server.py`               | Uses factory; logs active backend         |
| `benchmarks/bench_stores.py`      | Benchmark runner → `results.json`         |
| `benchmarks/results.json`         | Raw benchmark output                      |
| `dashboard.html`                  | Self-contained interactive dashboard      |
| `COMPARISON.md`                   | This file                                 |
| `requirements.txt`                | `redis`, `requests`, `pymongo`, `dnspython` |
