# Local LLM Chatbot System

A single-user, local-first chatbot that runs entirely on your own machine. The
system is a small async TCP server that proxies chat messages to a local
**Ollama** instance (any model you have pulled — `qwen3:8b` by default in this
checkout, with `gemma3:4b` and `llama3.1` as easy alternatives), and persists
conversation history in a pluggable store. Three clients are included: a
**polished terminal UI** with slash commands, colored Markdown panels, and
token streaming (`chatbot.cli`, recommended); a **bare-bones TCP REPL**
(`chatbot.clients.tcp_client`, legacy); and the **TCP server itself** for
custom integrations. A self-contained HTML dashboard visualizes side-by-side
benchmarks of two store backends: **Redis** and **MongoDB Atlas M0 (free
tier)**.

> Designed for a desktop with an **NVIDIA RTX 5060 (Blackwell/GB206, 8 GB
> VRAM)**. Total codebase: ~1,000 lines across 12 Python files + 1 HTML
> file + 30 unit tests.

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Running the chatbot](#5-running-the-chatbot)
6. [Using the Terminal UI (recommended)](#6-using-the-terminal-ui-recommended)
7. [Slash command reference](#7-slash-command-reference)
8. [Switching models](#8-switching-models)
9. [Switching store backends](#9-switching-store-backends)
10. [Verifying GPU usage](#10-verifying-gpu-usage)
11. [Running the test suite](#11-running-the-test-suite)
12. [Running the benchmark suite](#12-running-the-benchmark-suite)
13. [Viewing the comparison dashboard](#13-viewing-the-comparison-dashboard)
14. [Project layout](#14-project-layout)
15. [How a request flows](#15-how-a-request-flows)
16. [Configuration reference](#16-configuration-reference)
17. [Troubleshooting](#17-troubleshooting)
18. [Conclusion — Redis vs MongoDB](#18-conclusion--redis-vs-mongodb)

---

## 1. What it does

```
You ──▶ Terminal UI  ──▶ ┐
     ──▶ TCP client  ──▶ ├──▶ asyncio server ──▶ Ollama (qwen3:8b) ──▶ response
     ──▶ your code   ──▶ ┘           │
                                    ├──▶ Redis / MongoDB / in-memory (history)
                                    └──▶ logs/server.log (rotating)
```

- **Single TCP port** (default 9000) accepts newline-terminated JSON requests.
- **One Ollama call per turn**, with optional **token streaming** in the TUI
  (the assistant panel updates in place as Ollama emits tokens).
- **History is persisted** between turns and trimmed to the last 20 messages.
- **Special `__clear__` message** wipes history for the user.
- **Logs** rotate at 1 MB × 3 backups, mirrored to stdout.
- **Three store backends** work transparently — pick at startup via env var
  or CLI flag; the transport auto-degrades to in-memory if the chosen
  backend is unreachable.

A complete multi-turn conversation that demonstrates context retention:

```
you> My favorite color is blue. Please remember it.
bot> Okay, blue it is! 😊
you> What is my favorite color?
bot> Your favorite color is blue. 😊
you> /clear
  · history cleared
you> What is my favorite color?
bot> As an AI, I have no memory of past conversations or your personal preferences …
```

---

## 2. Architecture

```
                     ┌────────────────────────────┐
   TCP client        │  asyncio TCP server        │         ┌──────────────┐
   (clients/         │  chatbot/server.py         │ ──────▶ │   Ollama     │
    tcp_client.py)   │  port 127.0.0.1:9000       │ ◀────── │  qwen3:8b    │
        │            │                            │         │  (GPU 100%)  │
        │            │  ┌──────────────────────┐  │         └──────────────┘
        └──────────▶ │  │ Store (via factory)  │  │
                     │  │   STORE_BACKEND =    │  │
                     │  │ redis|mongo|memory   │  │
                     │  └──────────┬───────────┘  │
                     └─────────────┼──────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   ┌─────────────────┐   ┌──────────────────────┐   ┌────────────┐
   │   Redis 7       │   │  MongoDB 7 / Atlas   │   │ MemoryStore│
   │   key:chat:uid  │   │  M0 (history[])      │   │ (in-proc,  │
   │   value: JSON[] │   │                      │   │  no deps)  │
   └─────────────────┘   └──────────────────────┘   └────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                  Terminal UI  (chatbot.cli)                     │
   │  Rich panels · prompt_toolkit completion · slash commands ·    │
   │  token streaming · in-process by default · optional --tcp      │
   └─────────────────────────────────────────────────────────────────┘
```

The `chatbot/store.py` module defines a `Store` Protocol with three methods
(`get_history`, `save_history`, `clear_history`) and a `make_store()` factory
that returns a `RedisStore`, `MongoStore`, or `MemoryStore` based on
`STORE_BACKEND`. Both the server and the in-process transport use the same
factory; neither knows or cares which one it gets.

The TUI sits alongside the server as a peer entry point. It uses the same
`build_prompt` / Ollama HTTP call as the server, but in-process — no
socket, no port, no JSON line protocol.

---

## 3. Prerequisites

You need:

| Tool | Version | Why |
|------|---------|-----|
| **Python** | 3.11+ | Async TCP server, type hints |
| **Ollama** | 0.6+ (1.x is fine) | LLM runtime; **0.6+ required for Blackwell / RTX 5060 GPU support** |
| **NVIDIA GPU** | RTX 5060 / Blackwell-class, 8 GB VRAM | Runs qwen3:8b (~5.2 GB loaded) or gemma3:4b (~3.3 GB) |
| **NVIDIA driver** | CUDA 12.8+ | Required by Ollama 0.6+ for Blackwell |
| **Redis** (backend #1, optional) | 7+ | Local or containerized |
| **MongoDB** (backend #2, optional) | 7+ local, **or** MongoDB Atlas M0 free tier | Local or cloud |
| **Docker** (optional) | any recent | Easiest way to run Redis and/or MongoDB locally |

> **Don't have Redis or MongoDB?** The TUI works without either: pass
> `--memory` for a session-only store, or just start it without flags and
> let it auto-degrade. See [§6.3](#63-store-backends-and-auto-degrade).

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
rich>=13.7
prompt_toolkit>=3.0.43
```

The first four are needed for the server; the last two (`rich`,
`prompt_toolkit`) are needed only for the TUI. `pyperclip` is optional
(used by `/copy`).

> `pymongo` and `dnspython` are only required for the MongoDB backend. If
> you only use Redis or `--memory`, you can skip them: `pip install redis
> requests rich prompt_toolkit`.

### 4.2. Install and start Ollama

```bash
# Install from https://ollama.com/download (Windows installer / macOS .dmg /
# Linux curl install script)

# Pull the default model (~5.2 GB download, one-time)
ollama pull qwen3:8b

# Start the server (leave running in its own terminal, or run in background)
ollama serve
```

Verify it's up:
```bash
curl http://localhost:11434/api/tags
# should list "qwen3:8b" in the models array
```

See [§8](#8-switching-models) for using other models.

### 4.3. Install Redis (optional backend)

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

### 4.4. Install MongoDB (optional backend)

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
6. Test the URI by setting it as an env var (see [§9](#9-switching-store-backends))

---

## 5. Running the chatbot

There are three ways to chat. Pick one:

### 5.1. Terminal UI (recommended)

```bash
python -m chatbot.cli
```

That single command starts the chat in-process, prints a styled banner,
and drops you at the prompt. See [§6](#6-using-the-terminal-ui-recommended)
for the full walkthrough.

### 5.2. Server + bare TCP client

In one terminal:
```bash
python -m chatbot.server
```

In another:
```bash
python -m chatbot.clients.tcp_client
```

You'll be prompted for a `user_id` once, then can chat. The bare client
only recognises `clear` and `quit` — for the full slash-command system,
use the TUI in [§5.1](#51-terminal-ui-recommended).

### 5.3. Server only (custom client)

```bash
python -m chatbot.server
```

Then talk to `127.0.0.1:9000` with any TCP client that can send one
newline-terminated JSON line per request:

```python
import json, socket

s = socket.create_connection(("127.0.0.1", 9000))
s.sendall((json.dumps({"user_id": "alice", "message": "Hello"}) + "\n").encode())
reply = json.loads(s.makefile("rb").readline().decode().strip())
print(reply["response"])
s.close()
```

The server confirms it's listening with a log line like:

```
2026-06-15 22:10:02,887 INFO chatbot.server Chatbot server listening on 127.0.0.1:9000 (backend=redis)
```

The same wire format works against every store backend — no client-side
change needed. To stop: `Ctrl+C`.

---

## 6. Using the Terminal UI (recommended)

The polished REPL lives in `chatbot.cli`. It uses
[Rich](https://github.com/Textualize/rich) for colored panels and Markdown,
and [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)
for history, multi-line input, and slash-command completion.

### 6.1. Launch

```bash
python -m chatbot.cli              # in-process transport (default)
python -m chatbot.cli --memory     # no Redis/Mongo needed (session-only)
python -m chatbot.cli --tcp        # talk to a running server
python -m chatbot.cli --user alice # pick a user id non-interactively
python -m chatbot.cli --no-banner  # skip the startup banner
```

The first launch looks like this:

```
  _      __    __                     __          __
 | | /| / /__ / /  ___ ____  ___ ____/ /  ___ ___/ /
 | |/ |/ / -_) _ \/ _ `/ _ \/ _ `/ _  /  / _ `/ _  /
 |__/|__/\__/_.__/\_,_/_//_/\_,_/\_,_/   \_,_/\_,_/

 ╭────────── session ──────────╮
 │ backend  in-process         │
 │ user     alice              │
 │ model    qwen3:8b           │
 │ stream   on                 │
 ╰─────────────────────────────╯
 type /help for commands, /quit to leave.

you ▸ hi
 ╭─ you ─────────────────────────────────╮
 │ hi                                    │
 ╰───────────────────────────────────────╯
 ╭─ assistant ───────────────────────────╮
 │ Hello! How can I help you today?      │
 ╰───────────────────────────────────────╯
```

### 6.2. Switching models

Three equivalent ways to use a different Ollama model:

```bash
# 1. Per-launch env var (PowerShell)
$env:MODEL_NAME="gemma3:4b"; python -m chatbot.cli

# 2. Per-launch env var (bash)
MODEL_NAME=gemma3:4b python -m chatbot.cli

# 3. Mid-session slash command (no restart)
you ▸ /model llama3.1
you ▸ /model
llama3.1
```

See [§8](#8-switching-models) for the full list of options.

### 6.3. Store backends and auto-degrade

The TUI picks the same store as the server. Three options:

| Flag / setting | What you get |
| --- | --- |
| (default) | `STORE_BACKEND` from env, or `redis` if unset. Probes at startup; if the backend is unreachable, prints a one-line notice and continues. |
| `--memory` | Forces an in-process `MemoryStore`. History is session-only and lost on exit. No external services needed. |
| `--tcp` | Server is authoritative for history; the client doesn't touch the store. |

**Auto-degrade** (default mode): if `STORE_BACKEND=redis` and Redis is down,
the very first store call will fail. The transport catches it, swaps to
`MemoryStore`, prints:

```
  · history store unavailable — falling back to in-memory (history will not persist)
```

…and keeps going. No crash, no stack trace.

**Startup probe**: if the configured backend is unreachable, you'll see a
warning before the prompt appears:

```
  · history store probe failed (ConnectionError: ...) — rerun with --memory to skip persistence, or start Redis/Mongo.
```

### 6.4. Highlights

- Colored Markdown panels for every turn (cyan for `you`, green for the
  assistant, red for errors, dim italic for notices).
- **Token streaming** in in-process mode — the assistant panel updates
  in place as Ollama emits tokens, with a `▍` cursor during generation.
- Up-arrow recalls prior inputs; input history is persisted to
  `~/.chatbot_cli_history`.
- `<Tab>` completes slash commands and aliases.
- `Ctrl-C` continues to a new prompt; `Ctrl-D` exits cleanly.

### 6.5. Verifying it works

```bash
# 1. Start Ollama in one terminal
ollama serve

# 2. In another terminal, launch the TUI
python -m chatbot.cli --memory
# (--memory sidesteps Redis/Mongo entirely for the smoke test)

# 3. Type a message
you ▸ /model qwen3:8b
you ▸ Hello!

# 4. Try slash commands
you ▸ /help
you ▸ /history
you ▸ /temp 0.2
you ▸ /quit
```

---

## 7. Slash command reference

Type `/` and press <kbd>Tab</kbd> to complete. Aliases are shown in
parentheses.

| Command | What it does |
| --- | --- |
| `/help [cmd]` | List all commands, or show detail for one. |
| `/clear` (`/cls`) | Wipe the current history. |
| `/quit` (`/exit`) | Leave the chat. |
| `/history [n]` | Show the last *n* turns (default 10). |
| `/model [name]` | Show or set the model (override, not persisted to `config.py`). |
| `/system [text]` | Show or set the system prompt. |
| `/temp <0.0-2.0>` (`/temperature`) | Show or set sampling temperature. |
| `/ctx <int>` (`/num_ctx`) | Show or set the context window size. |
| `/save <name>` | Snapshot current history under a name. |
| `/load <name>` | Restore a named snapshot. |
| `/user <id>` | Switch user (loads that user's history). |
| `/new` | Fresh empty history for the current user. |
| `/stream [on\|off]` | Toggle token streaming (in-process only). |
| `/copy` | Copy the last reply to the clipboard (needs `pyperclip`). |
| `/retry` | Re-send the last user message, regenerating the reply. |

**Per-command detail via `/help <name>`:**

```
you ▸ /help temp
╭─ / temp ──────────────────────────────╮
│ usage    /temp <0.0-2.0>              │
│ aliases  /temperature                 │
│ summary  Show or set the sampling      │
│          temperature (0.0-2.0).        │
╰───────────────────────────────────────╯
```

**Validation:** `/temp 5.0` is rejected (out of range); `/ctx 16` is
rejected (minimum 128); `/model` with no args prints the current model
without changing it.

---

## 8. Switching models

### 8.1. Per-session slash command

```
you ▸ /model qwen3:8b
you ▸ /model llama3.1
you ▸ /model gemma3:4b
```

The change applies to the next turn. The banner reflects the current
model on redraw.

### 8.2. Per-launch env var

```bash
# PowerShell
$env:MODEL_NAME="gemma3:4b"; python -m chatbot.cli

# bash / Git Bash
MODEL_NAME=gemma3:4b python -m chatbot.cli
```

The env var is read once at import time, so it must be set **before**
launching Python. For runtime swaps, use `/model`.

### 8.3. Project-wide default

Edit `chatbot/config.py`:

```python
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:8b")
```

This affects every entry point (server, TCP client, TUI) and is what
`/model` falls back to when there's no override.

### 8.4. Pull first

Whichever model you pick must be pulled into Ollama first:

```bash
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull llama3.1
```

---

## 9. Switching store backends

The backend is selected at startup via the `STORE_BACKEND` env var or
the `--memory` / `--tcp` TUI flags. There is no need to edit any code.

### 9.1. Redis (default)

```bash
python -m chatbot.server
# or explicitly:
STORE_BACKEND=redis python -m chatbot.server
```

Requires Redis on `localhost:6379` (override via `REDIS_HOST` / `REDIS_PORT`
in `chatbot/config.py`).

### 9.2. MongoDB local

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

### 9.3. MongoDB Atlas M0 (free tier)

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

### 9.4. In-memory (no backend)

```bash
python -m chatbot.cli --memory
```

Uses a process-local `MemoryStore` from `chatbot/memory_store.py`. History
is lost on exit. No external services required.

Or set `STORE_BACKEND=memory` for the server (similarly transient).

---

## 10. Verifying GPU usage

After the first message has been processed, in a **separate terminal**:

```bash
ollama ps
```

You should see something like:

```
NAME         ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3:8b     500a1f067a9f    5.5 GB    100% GPU     4096       4 minutes from now
```

**Key column:** `PROCESSOR` should read **`100% GPU`**.

If it reads `100% CPU`:
1. Check Ollama version: `ollama --version` (must be 0.6+)
2. Update your NVIDIA driver to one with **CUDA 12.8+** support
3. On Linux, confirm `nvidia-smi` works and reports your GPU
4. Restart `ollama serve` and re-send a message

---

## 11. Running the test suite

The TUI command registry has 30 unit tests covering parsing, dispatch,
alias resolution, validation, and session state.

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_cli_commands.py::test_is_command_accepts_known_names PASSED
...
30 passed in 0.5s
```

The tests use a `FakeTransport` that records every call in memory — no
Redis, no Mongo, no Ollama required. Add new commands and tests in the
same module.

---

## 12. Running the benchmark suite

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

## 13. Viewing the comparison dashboard

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

## 14. Project layout

```
Local-LLM-Chatbot-System/
├── README.md                     ← you are here
├── COMPARISON.md                 ← detailed Redis vs MongoDB writeup
├── dashboard.html                ← interactive comparison dashboard
├── requirements.txt
├── chatbot/                      ← all source code
│   ├── __init__.py
│   ├── config.py                 ← constants (no logic)
│   ├── store.py                  ← Store Protocol + make_store() + probe_store()
│   ├── memory_store.py           ← in-process store (no external deps)
│   ├── redis_store.py            ← RedisStore class + module-level fns
│   ├── mongo_store.py            ← MongoStore class
│   ├── llm.py                    ← build_prompt() + call_ollama()
│   ├── server.py                 ← async TCP server entry point
│   ├── cli.py                    ← TUI entry point (recommended)
│   ├── clients/
│   │   └── tcp_client.py         ← bare-bones REPL (legacy)
│   └── tui/                      ← TUI implementation
│       ├── __init__.py
│       ├── session.py            ← Session dataclass (state, overrides)
│       ├── transport.py          ← Transport Protocol, InProcess, TCP
│       ├── commands.py           ← 15 slash commands, dispatch, parsing
│       ├── render.py             ← rich panels, banner, streaming Live
│       └── completer.py          ← prompt_toolkit WordCompleter
├── benchmarks/
│   ├── bench_stores.py           ← benchmark runner
│   └── results.json              ← generated benchmark output
├── tests/                        ← pytest unit tests
│   ├── __init__.py
│   └── test_cli_commands.py      ← 30 tests for the TUI command registry
├── logs/                         ← rotating server logs (created at runtime)
│   └── server.log
└── .gitignore
```

**Total Python code: ~1,000 lines across 12 files. The whole project is
small enough to read end-to-end in one sitting, and the TUI is fully
isolated in `chatbot/tui/` so you can read it independently.**

---

## 15. How a request flows

### 15.1. In-process TUI (default)

1. **You** type a message at the `you ▸ ` prompt.
2. **`chatbot/cli.py:_send_user_message`** pulls current history from
   the transport, calls `transport.send()` (or `transport.stream()` if
   streaming is on).
3. **`chatbot/tui/transport.py:InProcessTransport`** builds the prompt
   with `llm.build_prompt` (or its override-aware variant), then
   `requests.post(OLLAMA_URL, json={stream: True/False})`.
4. **Ollama** runs the model on the GPU and returns the response
   (token-by-token for streaming, single blob otherwise).
5. The TUI renders the reply in a green Markdown panel.
6. The transport appends the turn to history and `save_history`s
   through the configured `Store`.

### 15.2. Server + client

1. **Client** sends one newline-terminated JSON line:
   `{"user_id": "alice", "message": "Hello"}\n`
2. **Server** (`chatbot/server.py`) reads the line, validates it.
3. If `message == "__clear__"` → call `store.clear_history(user_id)`,
   send `{"response": "History cleared."}`, close.
4. Otherwise:
   - `history = store.get_history(user_id)`  (returns `[]` on first turn)
   - `prompt = llm.build_prompt(history, message)`
   - `response = llm.call_ollama(prompt)`  (HTTP POST to
     `http://localhost:11434/api/generate` with `stream: false`)
   - Append both turns to history, `store.save_history(user_id, history)`
5. **Server** sends `{"response": "<assistant text>"}\n` and closes.
6. **Log** records `user_id`, `msg_len`, `resp_len`, `latency_ms`.

If Ollama is unreachable, both paths catch the exception and surface a
clean error message (a red panel in the TUI; a `{"error": "..."}` JSON
line from the server).

---

## 16. Configuration reference

All configuration lives in `chatbot/config.py` (constants) and a small
set of env vars.

| Setting | Default | Purpose |
|---------|---------|---------|
| `HOST` | `127.0.0.1` | TCP bind address (localhost only) |
| `PORT` | `9000` | TCP port |
| `STORE_BACKEND` (env) | `redis` | `redis`, `mongo`, or `memory` |
| `MONGO_URI` (env) | *(empty)* | Required if `STORE_BACKEND=mongo` |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis logical DB |
| `MONGO_DB` | `chatbot` | Mongo database name |
| `MONGO_COLLECTION` | `history` | Mongo collection name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama HTTP endpoint |
| `MODEL_NAME` (env) | `qwen3:8b` | Default Ollama model tag (overridable at runtime via `/model`) |
| `MAX_MESSAGES` | `20` | History trim length |
| `SYSTEM_PROMPT` | `"You are a helpful assistant."` | System line of every prompt (overridable via `/system`) |
| `NUM_GPU` | `99` | Ollama `num_gpu` (99 = offload everything; Ollama auto-caps) |
| `NUM_CTX` | `4096` | Context window in tokens (overridable via `/ctx`) |
| `TEMPERATURE` | `0.7` | Sampling temperature (overridable via `/temp`) |
| `REPEAT_PENALTY` | `1.1` | Ollama `repeat_penalty` |

`NUM_GPU=99` is safe — Ollama clamps to the actual layer count. `NUM_CTX=4096`
fits comfortably in 8 GB VRAM with qwen3:8b (~5.2 GB loaded, ~3 GB headroom
for KV cache) or gemma3:4b (~3.3 GB loaded, ~4.5 GB headroom).

---

## 17. Troubleshooting

### "LLM unavailable"
- Is `ollama serve` running? Check `curl http://localhost:11434/`.
- Is the model pulled? `ollama list` should show `qwen3:8b` (or whichever
  model you've configured).

### Server won't start — "Address already in use"
- Something is on port 9000. Change `PORT` in `chatbot/config.py`, or kill
  the other process.

### TUI crashes on first message with a Redis ConnectionError
- The transport should auto-degrade. If you still see a stack trace, please
  file a bug with the output of `python -m chatbot.cli --no-banner`.
- Workaround: `python -m chatbot.cli --memory` skips Redis entirely.

### Redis backend fails on first call
- Is Redis running? `redis-cli ping` should return `PONG`.
- Default config points to `localhost:6379`. If your Redis is elsewhere,
  edit `REDIS_HOST`/`REDIS_PORT` in `chatbot/config.py`.

### Mongo backend fails — "MONGO_URI is empty"
- You forgot to set the env var. See [§9.2 / §9.3](#9-switching-store-backends).

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
  `tcp_client.py` (which sets UTF-8 stdout automatically). The TUI handles
  this for you.

### History seems to reset between turns
- Check `logs/server.log` for `save_history` calls. If you see "Redis
  connection refused", the store is silently falling back to an empty
  list. Fix the backend connection, or use `--memory` if persistence
  isn't important.

### `/copy` says "pyperclip is not installed"
- Optional dep: `pip install pyperclip`. On Linux you may also need
  `apt install xclip` or `apt install xsel`.

---

## 18. Conclusion — Redis vs MongoDB

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

1. **Chat latency is dominated by inference, not storage.** qwen3:8b takes
   **~2-4 s** to generate a response. Adding 30 ms (or even 100 ms) for a
   Mongo round-trip is **< 5%** of total turn time. The user cannot tell
   the difference between 2.40 s and 2.43 s.

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
- You're spinning up the TUI on a fresh machine with no time to set up a
  store — in that case, **`--memory` is the better choice over Redis**.

### Final recommendation

> **Use MongoDB Atlas M0** for this chatbot. Set `STORE_BACKEND=mongo` and
> `MONGO_URI` to your Atlas SRV string. You'll get durable, queryable
> history, zero local-ops overhead, and latency that is invisible behind
> the 2–4 second LLM inference time. For quick local hacking, use
> `--memory` instead — same code path, no setup.

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
