# Local LLM Chatbot System

> A fully offline, privacy-first conversational AI — TCP server + Redis + Ollama + Gemma 3 4B running on your RTX 5060.

---

## Table of Contents

1. [What It Is](#what-it-is)
2. [System Architecture](#system-architecture)
3. [Request Flow](#request-flow)
4. [Components](#components)
5. [File Structure](#file-structure)
6. [Features](#features)
7. [GPU Configuration](#gpu-configuration)
8. [Conversation Memory](#conversation-memory)
9. [Context Window Management](#context-window-management)
10. [Error Handling](#error-handling)
11. [Logging](#logging)
12. [Scalability Roadmap](#scalability-roadmap)

---

## What It Is

A locally-hosted chatbot that runs entirely on your machine. No cloud APIs, no data leaving your network, no rate limits, no per-token billing. You send a message over a TCP socket, it fetches your conversation history from Redis, builds a prompt, runs it through Gemma 3 4B on your GPU, and sends back a response.

```
Your terminal ──► TCP socket ──► Python server ──► Redis + Ollama ──► back to you
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Your Machine                           │
│                                                                 │
│   ┌──────────────┐         ┌──────────────────────────────────┐ │
│   │  TCP Client  │         │         Python Server            │ │
│   │              │ ──────► │       (AsyncIO / port 9000)      │ │
│   │ tcp_client   │ JSON    │                                  │ │
│   │    .py       │ ◄────── │  • validates payload             │ │
│   └──────────────┘         │  • builds prompt                 │ │
│                            │  • calls Ollama                  │ │
│                            │  • saves history                 │ │
│                            └──────────┬───────────────────────┘ │
│                                       │                         │
│                        ┌──────────────┼──────────────┐          │
│                        │              │              │          │
│                        ▼              ▼              │          │
│               ┌────────────┐  ┌─────────────┐       │          │
│               │   Redis    │  │   Ollama    │       │          │
│               │            │  │             │       │          │
│               │ chat:user1 │  │ localhost   │       │          │
│               │ chat:user2 │  │   :11434    │       │          │
│               │ chat:user3 │  │             │       │          │
│               └────────────┘  └──────┬──────┘       │          │
│                                      │              │          │
│                                      ▼              │          │
│                             ┌─────────────────┐     │          │
│                             │   RTX 5060 8GB  │     │          │
│                             │                 │     │          │
│                             │  Gemma 3 4B     │     │          │
│                             │  ~3.3GB VRAM    │     │          │
│                             │  num_gpu: 99    │     │          │
│                             └─────────────────┘     │          │
└─────────────────────────────────────────────────────┘          │
                                                                  │
```

---

## Request Flow

Every message goes through this exact sequence:

```
User types a message
        │
        ▼
tcp_client.py wraps it:
{"user_id": "alice", "message": "Explain neural networks"}
        │
        ▼ TCP socket (port 9000)
        │
┌───────▼──────────────────────────────────────────────┐
│                    server.py                          │
│                                                       │
│  1. Receive newline-terminated JSON                   │
│  2. Validate: user_id and message must be non-empty   │
│  3. Check for __clear__ magic message                 │
│                                                       │
│  4. redis_store.get_history("alice")                  │
│     └─► Redis key: chat:alice                         │
│         Returns: list of past {role, content} dicts   │
│                                                       │
│  5. llm.build_prompt(history, message)                │
│     └─► Assembles multi-turn plain-text prompt        │
│                                                       │
│  6. llm.call_ollama(prompt)                           │
│     └─► POST localhost:11434/api/generate             │
│         ├── model: gemma3:4b                          │
│         ├── stream: false                             │
│         └── options: num_gpu=99, num_ctx=4096         │
│                                                       │
│  7. Append both turns to history                      │
│  8. redis_store.save_history("alice", history)        │
│     └─► Trim to last 20 messages, SET in Redis        │
│                                                       │
│  9. Send {"response": "..."} back over TCP            │
│ 10. Log user_id, msg length, response length, latency │
└───────────────────────────────────────────────────────┘
        │
        ▼ TCP socket
        │
tcp_client.py prints the response
```

---

## Components

### `config.py` — Central Constants

All tuneable values live here. Change them in one place, everything picks them up.

| Constant | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `9000` | TCP port |
| `REDIS_HOST` | `localhost` | Redis address |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_NAME` | `gemma3:4b` | Ollama model |
| `MAX_MESSAGES` | `20` | History trim limit |
| `NUM_GPU` | `99` | GPU layers (auto-capped) |
| `NUM_CTX` | `4096` | Context window tokens |
| `TEMPERATURE` | `0.7` | Response randomness |
| `REPEAT_PENALTY` | `1.1` | Reduce repetition |
| `SYSTEM_PROMPT` | `"You are a helpful assistant."` | Base instruction |

---

### `redis_store.py` — Conversation Memory

```
Redis key structure
───────────────────
chat:alice   →  [ {role:user, content:...}, {role:assistant, content:...}, ... ]
chat:bob     →  [ {role:user, content:...}, {role:assistant, content:...}, ... ]
chat:carol   →  [ ... ]

Each value is a JSON-serialized list of {role, content} dicts.
Trimmed to the last MAX_MESSAGES entries on every write.
```

| Function | What it does |
|---|---|
| `get_history(user_id)` | Fetch and deserialize history. Returns `[]` if key missing. |
| `save_history(user_id, history)` | Trim to `MAX_MESSAGES`, serialize to JSON, SET in Redis. |
| `clear_history(user_id)` | DELETE the key entirely. |

---

### `llm.py` — Prompt Builder & Ollama Client

**Prompt format built by `build_prompt()`:**

```
System: You are a helpful assistant.

User: What is machine learning?
Assistant: Machine learning is a field of AI where systems learn from data...

User: Give me a real-world example.
Assistant: A good example is email spam filtering...

User: <new message goes here>
Assistant:
```

**Ollama API payload sent by `call_ollama()`:**

```json
{
  "model": "gemma3:4b",
  "prompt": "<built prompt>",
  "stream": false,
  "options": {
    "num_gpu": 99,
    "num_ctx": 4096,
    "temperature": 0.7,
    "repeat_penalty": 1.1
  }
}
```

---

### `server.py` — AsyncIO TCP Server

```
asyncio.start_server
        │
        ├── handle_client(reader, writer)   ← one coroutine per connection
        │         │
        │         ├── reader.readline()      ← async, non-blocking
        │         ├── json.loads()
        │         ├── validate payload
        │         ├── get history
        │         ├── build + call LLM
        │         ├── save history
        │         ├── writer.write()
        │         └── writer.close()
        │
        └── asyncio.run(main())
```

Connections are handled concurrently by the event loop — no threads, no blocking I/O.

---

### `clients/tcp_client.py` — CLI Test Client

```
$ python clients/tcp_client.py

Enter your user_id: alice

You: Explain transformers
Assistant: Transformers are a neural network architecture...

You: How do attention heads work?
Assistant: Each attention head learns to focus on different...

You: clear
→ History cleared.

You: Who are you?
Assistant: I'm a helpful assistant. How can I help you?
  (no memory of previous turns — history was reset)

You: quit
→ Bye.
```

**Special commands:**

| Input | What happens |
|---|---|
| Any text | Sends message, prints response |
| `clear` | Sends `__clear__`, wipes Redis history for your user_id |
| `quit` | Closes the TCP connection and exits |

---

## Features

### Multi-turn Conversation Memory

Each user gets an isolated Redis key. The server loads the full history on every request, injects it into the prompt, then saves the updated history back. The model sees the entire conversation each time, so it can reference earlier messages naturally.

```
Turn 1:  User: "My name is Arif."
         Assistant: "Nice to meet you, Arif!"

Turn 2:  User: "What's my name?"
         Assistant: "Your name is Arif."   ← remembered from Turn 1
```

---

### User Isolation

Each user's history is stored under a separate Redis key. `chat:alice` and `chat:bob` are completely independent — there is no way for one user's context to bleed into another's.

```
chat:alice  ──►  Alice's conversation only
chat:bob    ──►  Bob's conversation only
chat:carol  ──►  Carol's conversation only
```

---

### History Clear

Sending the magic message `__clear__` (via typing `clear` in the client) deletes the Redis key for that user. The next message starts a completely fresh conversation.

---

### GPU-Accelerated Inference

`num_gpu: 99` tells Ollama to offload every model layer to the GPU. Ollama auto-caps this to the actual layer count, so 99 is always safe and means "all layers."

```
Gemma 3 4B model weights    ~3.3 GB VRAM
KV cache for 4096 ctx        ~0.5–1.0 GB VRAM
──────────────────────────────────────────
Total used                   ~4.0–4.5 GB
RTX 5060 available           8.0 GB
Headroom                     ~3.5–4.0 GB  ✓
```

---

### Context Window Management

Without a limit, Redis history grows forever and eventually the prompt exceeds Gemma's context window, causing errors or truncation. `save_history()` always trims to the last `MAX_MESSAGES` (default 20) entries before writing.

```
History before trim (22 turns):
[turn1, turn2, turn3, ... turn20, turn21, turn22]
                                  ↑
                               trimmed
History after trim (20 turns):
[turn3, turn4, ... turn20, turn21, turn22]
```

20 messages = 10 back-and-forth exchanges, which covers most real sessions while keeping the prompt well within the 4096-token context window.

---

### Structured Logging

Every request writes a structured log line to both stdout and `logs/server.log`:

```
2026-06-15 14:32:01 INFO  user=alice  msg_len=34  resp_len=412  latency=1843ms
2026-06-15 14:32:44 INFO  user=alice  msg_len=28  resp_len=289  latency=1201ms
2026-06-15 14:33:10 ERROR user=bob    LLM unavailable
Traceback (most recent call last):
  ...
```

Log files rotate at 1 MB with 3 backups kept (`server.log`, `server.log.1`, `server.log.2`).

---

## GPU Configuration

The RTX 5060 uses NVIDIA's Blackwell architecture (GB206). Full GPU support requires:

| Requirement | Minimum version |
|---|---|
| CUDA | 12.8+ |
| Ollama | 0.6+ |
| Driver | 570+ (Blackwell) |

**Verify GPU is active** after sending your first message:

```bash
ollama ps
```

Expected output:
```
NAME           ID        SIZE    PROCESSOR    UNTIL
gemma3:4b      ...       4.9 GB  100% GPU     4 minutes from now
```

If it shows `100% CPU`, Ollama is falling back silently. Fix: update Ollama to 0.6+ and ensure CUDA 12.8 drivers are installed.

---

## Error Handling

```
Incoming message
      │
      ├── Missing user_id or message field?
      │         └──► {"error": "Invalid payload"}  →  close connection
      │
      ├── message == "__clear__"?
      │         └──► clear Redis key  →  {"response": "History cleared."}
      │
      ├── Ollama not running / times out?
      │         └──► {"error": "LLM unavailable"}  →  log full traceback
      │
      └── Redis not available?
                └──► exception propagates  →  logged, connection closed cleanly
```

The server never crashes on a bad client message. It always sends a valid JSON error response before closing the connection.

---

## Scalability Roadmap

```
V1 — Phase 1 (now)
─────────────────────────────────────────────
1 user · TCP · Redis · Gemma 3 4B · <500 lines


V2 — Multi-User (Phase 2)
─────────────────────────────────────────────
100 concurrent users
AsyncIO already handles this — just tune the
Redis connection pool and test under load.


V3 — Streaming (Phase 4)
─────────────────────────────────────────────
Set stream: true in Ollama payload.
Yield newline-delimited JSON tokens over TCP.
User sees output generated live.


V4 — Scale Out (future)
─────────────────────────────────────────────
Load balancer
    ├── TCP Server instance 1
    ├── TCP Server instance 2
    └── TCP Server instance 3
              │
         Redis Cluster
              │
         Ollama Cluster


V5 — RAG (future)
─────────────────────────────────────────────
Vector DB (ChromaDB)
    └── Document search → inject into prompt
             +
        Function calling
```

---

## Quick Reference

```bash
# 1. Start Redis
redis-server

# 2. Start Ollama
ollama serve

# 3. Pull the model (first time only)
ollama pull gemma3:4b

# 4. Start the chatbot server
python server.py

# 5. Open a client in another terminal
python clients/tcp_client.py

# 6. Verify GPU usage
ollama ps
```

---

*Phase 1 MVP · Local LLM Chatbot · June 2026*
