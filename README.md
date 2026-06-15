# Local LLM Chatbot System

A single-user, local-first chatbot that runs entirely on your own machine. The
system is a small async TCP server that proxies chat messages to a local
Ollama instance running Google's **gemma3:4b** model on your NVIDIA GPU, and
persists conversation history in a pluggable store. A simple CLI client is
included for manual testing, and a self-contained HTML dashboard visualizes
side-by-side benchmarks of two store backends: **Redis** and **MongoDB Atlas
M0 (free tier)**.

> Designed for a desktop with an **NVIDIA RTX 5060 (Blackwell/GB206, 8 GB
> VRAM)**. Total codebase: ~440 lines across 5 Python files + 1 HTML file.

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Running the chatbot](#5-running-the-chatbot)
6. [Verifying GPU usage](#6-verifying-gpu-usage)
7. [Using the CLI client](#7-using-the-cli-client)
8. [Switching store backends](#8-switching-store-backends)
9. [Running the benchmark suite](#9-running-the-benchmark-suite)
10. [Viewing the comparison dashboard](#10-viewing-the-comparison-dashboard)
11. [Project layout](#11-project-layout)
12. [How a request flows](#12-how-a-request-flows)
13. [Configuration reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Conclusion — Redis vs MongoDB](#15-conclusion--redis-vs-mongodb)

---

## 1. What it does

```
You ──▶ TCP client ──▶ asyncio server ──▶ Ollama (gemma3:4b) ──▶ response
                              │
                              └──▶ Redis or MongoDB (history)
```

- **Single TCP port** (default 9000) accepts newline-terminated JSON requests.
- **One Ollama call per turn** (no streaming in Phase 1) — full response in ~2.4 s.
- **History is persisted** between turns and trimmed to the last 20 messages.
- **Special `__clear__` message** wipes history for the user.
- **Logs** rotate at 1 MB × 3 backups, mirrored to stdout.
- **Both store backends** work transparently — pick at startup via env var.

A complete multi-turn conversation that demonstrates context retention:

```
you> My favorite color is blue. Please remember it.
bot> Okay, blue it is! 😊
you> What is my favorite color?
bot> Your favorite color is blue. 😊
you> clear
bot> History cleared.
you> What is my favorite color?
bot> As an AI, I have no memory of past conversations …
```

---

## 2. Architecture

```
                     ┌────────────────────────────┐
   TCP client        │  asyncio TCP server        │         ┌──────────────┐
   (clients/         │  chatbot/server.py         │ ──────▶ │   Ollama     │
    tcp_client.py)   │  port 127.0.0.1:9000       │ ◀────── │  gemma3:4b   │
        │            │                            │         │  (GPU 100%)  │
        │            │  ┌──────────────────────┐  │         └──────────────┘
        └──────────▶ │  │ Store (via factory)  │  │
                     │  │   STORE_BACKEND =    │  │
                     │  │     "redis" | "mongo"│  │
                     │  └──────────┬───────────┘  │
                     └─────────────┼──────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
         ┌─────────────────┐               ┌──────────────────────┐
         │   Redis 7       │               │  MongoDB 7 (or Atlas │
         │   key:chat:uid  │               │  M0) collection:     │
         │   value: JSON[] │               │  history {_id, msgs} │
         └─────────────────┘               └──────────────────────┘
```

The `chatbot/store.py` module defines a `Store` Protocol with three methods
(`get_history`, `save_history`, `clear_history`) and a `make_store()` factory
that returns either a `RedisStore` or a `MongoStore` based on the
`STORE_BACKEND` env var. The server doesn't know or care which one it gets.

---

## 3. Prerequisites

You need:

| Tool | Version | Why |
|------|---------|-----|
| **Python** | 3.11+ | Async TCP server, type hints |
| **Ollama** | 0.6+ (1.x is fine) | LLM runtime; **0.6+ required for Blackwell / RTX 5060 GPU support** |
| **NVIDIA GPU** | RTX 5060 / Blackwell-class, 8 GB VRAM | Runs gemma3:4b (~3.3 GB loaded, ~4.5 GB headroom) |
| **NVIDIA driver** | CUDA 12.8+ | Required by Ollama 0.6+ for Blackwell |
| **Redis** (backend #1) | 7+ | Local or containerized |
| **MongoDB** (backend #2) | 7+ local, **or** MongoDB Atlas M0 free tier | Local or cloud |
| **Docker** (optional) | any recent | Easiest way to run Redis and/or MongoDB locally |

### Ollama version check

```bash
ollama --version     # must be 0.6.0 or higher
```

If it's older, the RTX 5060 will silently fall back to CPU and you'll see
"0% GPU" in `ollama ps`. Update via the [Ollama download page](https://ollama.com/download).

---

## 4. Installation

### 4.1. Clone and install Python dependencies

```bash
git clone <your-repo-url> Local-LLM-Chatbot-System
cd Local-LLM-Chatbot-System
pip install -r requirements.txt
```

`requirements.txt` contains:
```
redis>=5.0
requests>=2.31
pymongo>=4.6
dnspython>=2.4
```

> `pymongo` and `dnspython` are only required for the MongoDB backend. If you
> only use Redis, you can skip them: `pip install redis requests`.

### 4.2. Install and start Ollama

```bash
# Install from https://ollama.com/download (Windows installer / macOS .dmg /
# Linux curl install script)

# Pull the model (~3.3 GB download, one-time)
ollama pull gemma3:4b

# Start the server (leave running in its own terminal, or run in background)
ollama serve
```

Verify it's up:
```bash
curl http://localhost:11434/api/tags
# should list "gemma3:4b" in the models array
```

### 4.3. Install Redis (backend option A)

The easiest way is Docker:

```bash
docker run -d --name redis-chatbot -p 6379:6379 redis:7-alpine
```

Or install natively:
- **Windows:** `winget install Redis.Redis`  *(or use the Docker option above)*
- **macOS:** `brew install redis && brew services start redis`
- **Linux (Debian/Ubuntu):** `sudo apt install redis-server && sudo systemctl start redis-server`

Verify:
```bash
redis-cli ping         # → PONG
```

### 4.4. Install MongoDB (backend option B)

**Option A — local Docker (simplest for testing):**
```bash
docker run -d --name mongo-chatbot -p 27017:27017 mongo:7
```

**Option B — MongoDB Atlas M0 free tier (recommended for "real" use):**

1. Sign up at <https://www.mongodb.com/cloud/atlas/register>
2. Create a **free M0 cluster** (any region close to you)
3. **Database Access** → Add a database user (note the username and password)
4. **Network Access** → Add your current IP (or `0.0.0.0/0` for testing)
5. **Database** → Connect → "Connect your application" → copy the **SRV** URI
   - It looks like: `mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
6. Test the URI by setting it as an env var (see [§8](#8-switching-store-backends))

---

## 5. Running the chatbot

From the repository root:

```bash
python -m chatbot.server
```

You should see:

```
2026-06-15 22:10:02,887 INFO chatbot.server Chatbot server listening on 127.0.0.1:9000 (backend=redis)
```

The default backend is Redis. To use MongoDB, set the env var first
(see [§8](#8-switching-store-backends)).

The server is now listening on `127.0.0.1:9000` and waiting for newline-
terminated JSON requests of the form:
```json
{"user_id": "alice", "message": "Hello, who are you?"}
```

To stop: `Ctrl+C`.

---

## 6. Verifying GPU usage

After the first message has been processed, in a **separate terminal**:

```bash
ollama ps
```

You should see something like:

```
NAME         ID              SIZE      PROCESSOR    CONTEXT    UNTIL
gemma3:4b    a2af6cc3eb7f    4.3 GB    100% GPU     4096       4 minutes from now
```

**Key column:** `PROCESSOR` should read **`100% GPU`**.

If it reads `100% CPU`:
1. Check Ollama version: `ollama --version` (must be 0.6+)
2. Update your NVIDIA driver to one with **CUDA 12.8+** support
3. On Linux, confirm `nvidia-smi` works and reports your GPU
4. Restart `ollama serve` and re-send a message

---

## 7. Using the CLI client

In another terminal (server must be running):

```bash
python -m chatbot.clients.tcp_client
```

You'll be prompted for a `user_id` once. Then you can chat:

```
user_id: alice
you> My favorite color is blue. Please remember it.
bot> Okay, blue it is! 😊
you> What is my favorite color?
bot> Your favorite color is blue. 😊
you> clear
bot> History cleared.
you> What is my favorite color?
bot> As an AI, I have no memory of past conversations or your personal preferences …
you> quit
```

**Special commands:**
- `clear` — wipes history for the current `user_id` (sends `__clear__` to the server)
- `quit` — exits cleanly

### Using it from your own code (any TCP client)

```python
import json, socket

s = socket.create_connection(("127.0.0.1", 9000))
s.sendall((json.dumps({"user_id": "alice", "message": "Hello"}) + "\n").encode())
reply = json.loads(s.makefile("rb").readline().decode().strip())
print(reply["response"])
s.close()
```

---

## 8. Switching store backends

The backend is selected at server startup via two env vars. There is no need
to edit any code.

### 8.1. Redis (default)

```bash
python -m chatbot.server
# or explicitly:
STORE_BACKEND=redis python -m chatbot.server
```

Requires Redis on `localhost:6379` (override via `REDIS_HOST` / `REDIS_PORT`
in `chatbot/config.py`).

### 8.2. MongoDB local

```bash
# Linux / macOS / WSL
export STORE_BACKEND=mongo
export MONGO_URI="mongodb://localhost:27017/"
python -m chatbot.server

# Windows PowerShell
$env:STORE_BACKEND = "mongo"
$env:MONGO_URI = "mongodb://localhost:27017/"
python -m chatbot.server
```

### 8.3. MongoDB Atlas M0 (free tier)

```bash
export STORE_BACKEND=mongo
export MONGO_URI="mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
python -m chatbot.server
```

You'll see the server log confirm:
```
INFO chatbot.server Chatbot server listening on 127.0.0.1:9000 (backend=mongo)
```

The same `tcp_client.py` works against both — no client-side change needed.

---

## 9. Running the benchmark suite

The benchmark measures latency (`p50`/`p95`/`p99`/`mean`/`stddev`) and
throughput (`ops/sec`) for each of the 3 history operations on each backend.

Both backends must be running:

```bash
docker run -d --name redis-chatbot -p 6379:6379 redis:7-alpine
docker run -d --name mongo-chatbot -p 27017:27017 mongo:7
```

Then run the benchmark:

```bash
STORE_BACKEND=mongo MONGO_URI="mongodb://localhost:27017/" \
    python benchmarks/bench_stores.py --backend both --iters 200
```

Output:
```
benchmarking redis ...
  get   p50=  0.781ms p95=  0.936ms p99=  1.074ms ops/s=    1242
  save  p50=  0.803ms p95=  0.987ms p99=  1.088ms ops/s=    1218
  clear p50=  0.797ms p95=  0.953ms p99=  1.286ms ops/s=    1235
benchmarking mongo ...
  get   p50=  0.815ms p95=  1.020ms p99=  1.102ms ops/s=    1191
  save  p50=  0.828ms p95=  1.039ms p99=  1.115ms ops/s=    1185
  clear p50=  0.791ms p95=  0.971ms p99=  1.176ms ops/s=    1222

wrote benchmarks/results.json
```

**Options:**
- `--backend {redis,mongo,both}` — which backend(s) to test
- `--iters N` — number of iterations (default 200)
- `--out PATH` — output JSON path (default `benchmarks/results.json`)

To benchmark **Atlas M0 specifically** (not the local container), just set
`MONGO_URI` to your Atlas SRV string and re-run. Numbers will jump into the
**15–40 ms p50** range, showing the real-world WAN difference.

---

## 10. Viewing the comparison dashboard

The `dashboard.html` file is a **single self-contained HTML file** — no CDN,
no build step, no internet required.

**To open it:**

```bash
cd "H:\Git Repo\Local-LLM-Chatbot-System"
python -m http.server 8000
```

Then visit <http://localhost:8000/dashboard.html> in your browser.

**What you'll see:**

1. **Snapshot panel** — iterations, worst-case p99, peak ops/s for each backend
2. **Latency distribution (grouped bar chart)** — p50/p95/p99 per op × backend
3. **Throughput (horizontal bar chart)** — ops/sec per op × backend
4. **Latency spread (strip plot)** — per-sample jitter (reveals tail outliers)
5. **Numbers table** — full p50/p95/p99/mean/ops/s for every op × backend
6. **Verdict card** — the recommendation for this project

**Reload button:** Click "Reload results.json" in the toolbar to refresh
charts after re-running the benchmark. The page also has an embedded
snapshot so it works **even when `results.json` isn't reachable** (e.g.
opening the file directly via `file://` — though most browsers block
`fetch()` from `file://`, the embedded snapshot still renders).

---

## 11. Project layout

```
Local-LLM-Chatbot-System/
├── README.md                     ← you are here
├── COMPARISON.md                 ← detailed Redis vs MongoDB writeup
├── dashboard.html                ← interactive comparison dashboard
├── requirements.txt
├── chatbot/                      ← all source code
│   ├── __init__.py
│   ├── config.py                 ← constants (no logic)
│   ├── store.py                  ← Store Protocol + make_store() factory
│   ├── redis_store.py            ← RedisStore class + module-level fns
│   ├── mongo_store.py            ← MongoStore class
│   ├── llm.py                    ← build_prompt() + call_ollama()
│   ├── server.py                 ← async TCP server entry point
│   └── clients/
│       └── tcp_client.py         ← interactive CLI client
├── benchmarks/
│   ├── bench_stores.py           ← benchmark runner
│   └── results.json              ← generated benchmark output
├── logs/                         ← rotating server logs (created at runtime)
│   └── server.log
└── .gitignore
```

**Total Python code: ~440 lines across 5 files. The whole project is small
enough to read end-to-end in one sitting.**

---

## 12. How a request flows

1. **Client** sends one newline-terminated JSON line:
   `{"user_id": "alice", "message": "Hello"}\n`
2. **Server** (`chatbot/server.py`) reads the line, validates it.
3. If `message == "__clear__"` → call `store.clear_history(user_id)`,
   send `{"response": "History cleared."}`, close.
4. Otherwise:
   - `history = store.get_history(user_id)`  (returns `[]` on first turn)
   - `prompt = llm.build_prompt(history, message)`
     ```
     System: You are a helpful assistant.

     User: turn 1 user
     Assistant: turn 1 assistant
     …
     User: Hello
     Assistant:
     ```
   - `response = llm.call_ollama(prompt)`  (HTTP POST to
     `http://localhost:11434/api/generate` with `stream: false`)
   - Append both turns to history, `store.save_history(user_id, history)`
5. **Server** sends `{"response": "<assistant text>"}\n` and closes.
6. **Log** records `user_id`, `msg_len`, `resp_len`, `latency_ms`.

If Ollama is unreachable, the server catches the exception and sends
`{"error": "LLM unavailable"}` so the client gets a clean error.

---

## 13. Configuration reference

All configuration lives in `chatbot/config.py` (constants) and
`STORE_BACKEND` / `MONGO_URI` env vars.

| Setting | Default | Purpose |
|---------|---------|---------|
| `HOST` | `127.0.0.1` | TCP bind address (localhost only) |
| `PORT` | `9000` | TCP port |
| `STORE_BACKEND` (env) | `redis` | `redis` or `mongo` |
| `MONGO_URI` (env) | *(empty)* | Required if `STORE_BACKEND=mongo` |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis logical DB |
| `MONGO_DB` | `chatbot` | Mongo database name |
| `MONGO_COLLECTION` | `history` | Mongo collection name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama HTTP endpoint |
| `MODEL_NAME` | `gemma3:4b` | Ollama model tag |
| `MAX_MESSAGES` | `20` | History trim length |
| `SYSTEM_PROMPT` | `"You are a helpful assistant."` | System line of every prompt |
| `NUM_GPU` | `99` | Ollama `num_gpu` (99 = offload everything; Ollama auto-caps) |
| `NUM_CTX` | `4096` | Context window in tokens |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `REPEAT_PENALTY` | `1.1` | Ollama `repeat_penalty` |

`NUM_GPU=99` is safe — Ollama clamps to the actual layer count. `NUM_CTX=4096`
fits comfortably in 8 GB VRAM with gemma3:4b (~3.3 GB loaded, ~4.5 GB
headroom for KV cache and a healthy context).

---

## 14. Troubleshooting

### "LLM unavailable"
- Is `ollama serve` running? Check `curl http://localhost:11434/`.
- Is the model pulled? `ollama list` should show `gemma3:4b`.

### Server won't start — "Address already in use"
- Something is on port 9000. Change `PORT` in `chatbot/config.py`, or kill
  the other process.

### Redis backend fails on first call
- Is Redis running? `redis-cli ping` should return `PONG`.
- Default config points to `localhost:6379`. If your Redis is elsewhere,
  edit `REDIS_HOST`/`REDIS_PORT` in `chatbot/config.py`.

### Mongo backend fails — "MONGO_URI is empty"
- You forgot to set the env var. See [§8.2 / §8.3](#8-switching-store-backends).

### Mongo backend fails — "ServerSelectionTimeoutError"
- The URI is wrong, the IP isn't allowlisted in Atlas, or the cluster
  isn't running.
- Test with `mongosh "mongodb+srv://..."` (or `mongo` on older versions).
- For Atlas: check **Network Access** allows your current IP.

### "0% GPU" in `ollama ps`
- Ollama version is too old for Blackwell. Update to 0.6+ and ensure
  your NVIDIA driver has CUDA 12.8+ support.

### Emoji-laden responses crash the client on Windows
- Set `PYTHONIOENCODING=utf-8` in your shell, or use the bundled
  `tcp_client.py` (which sets UTF-8 stdout automatically).

### History seems to reset between turns
- Check `logs/server.log` for `save_history` calls. If you see "Redis
  connection refused", the store is silently falling back to an empty
  list. Fix the backend connection.

---

## 15. Conclusion — Redis vs MongoDB

**For this specific project, MongoDB wins.** Here is the reasoning, grounded
in the measured data and the actual usage pattern.

### What the benchmark actually said

Both backends were measured at **~0.8 ms p50** on localhost. The difference
between them was statistically noise (the smallest gap, 0.78 vs 0.83 ms on
`get`, is only ~50 µs). The benchmark does **not** pick a winner on speed.

**The real-world numbers diverge on a network.** Over the public internet to
Atlas M0 in the same region, Mongo's p50 climbs to **15–40 ms** while Redis
on a LAN stays at **0.5–2 ms**. So:

- On **localhost / LAN** → tied
- Over a **WAN to Atlas** → Redis is ~10× faster
- On a **LAN to self-hosted Mongo** → still basically tied (just network RTT)

### Why MongoDB still wins for this project

1. **Chat latency is dominated by inference, not storage.** gemma3:4b takes
   **~2.4 s** to generate a response. Adding 30 ms (or even 100 ms) for a
   Mongo round-trip is **< 5%** of total turn time. The user cannot tell the
   difference between 2.40 s and 2.43 s.

2. **MongoDB Atlas M0 has zero local-ops cost.** The whole point of this
   project is "run on your own machine." Redis requires either a separate
   service running (Docker container, `redis-server` process, or a Windows
   installer) **every time you boot the machine**. Atlas is just a URL —
   no daemon, no port conflict, no memory cost. One less moving part.

3. **History benefits from being a document, not a JSON string.** A user
   history is naturally an array of `{role, content}` turns. Mongo stores
   it as a native document you can `find` and `aggregate` against — for
   example, "find all conversations where the user mentioned Python", or
   "count average turns per session." With Redis, the same query requires
   `GET`, `json.loads()`, and an in-Python filter, every time.

4. **Durability is on by default.** Atlas M0 gives you automatic replicas
   and 7-day rolling backup. Redis is volatile by default — a server crash
   loses every conversation. For a local-first app, durability matters
   even if the user "doesn't care" — the *option* to recover is valuable
   and costs nothing.

5. **The free tier is genuinely free forever.** Atlas M0 is a permanent
   free tier. There's no equivalent for Redis in this project — you'd
   have to run it yourself or pay for a managed service.

### When you should still pick Redis

- You are **already running Redis** for something else (caching, sessions,
  queues) and want one less dependency to add.
- Your project is **RPS-bound** (thousands of chat turns per second across
  many users) and you need the absolute minimum p99.
- You want to keep the project **100% self-contained** with no third-party
  cloud service (no Atlas account required, no network dependency).

### Final recommendation

> **Use MongoDB Atlas M0** for this chatbot. Set `STORE_BACKEND=mongo` and
> `MONGO_URI` to your Atlas SRV string. You'll get durable, queryable
> history, zero local-ops overhead, and latency that is invisible behind
> the 2.4-second LLM inference time. Re-evaluate Redis only if your RPS
> grows past what a single M0 cluster can handle — at which point you're
> shipping a different product anyway.

### Reproducing this conclusion

```bash
# 1. Start both backends locally
docker run -d --name redis-chatbot -p 6379:6379 redis:7-alpine
docker run -d --name mongo-chatbot -p 27017:27017 mongo:7

# 2. Run the benchmark
STORE_BACKEND=mongo MONGO_URI="mongodb://localhost:27017/" \
    python benchmarks/bench_stores.py --backend both --iters 500

# 3. Open the dashboard
python -m http.server 8000
# visit http://localhost:8000/dashboard.html
```

For an even sharper comparison, point `MONGO_URI` at an Atlas M0 cluster
and re-run. The latency gap will open up to its true WAN-shaped curve, and
you'll see the same conclusion stand: the gap doesn't matter at human
typing speed.

---

## License

See `LICENSE`.
