"""Runtime constants for the local LLM chatbot."""

import os

HOST = "127.0.0.1"
PORT = 9000

# Store backend selector: "redis" (default) or "mongo"
STORE_BACKEND = os.environ.get("STORE_BACKEND", "redis")

# Redis connection (used when STORE_BACKEND="redis")
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# MongoDB connection (used when STORE_BACKEND="mongo")
# Atlas free tier SRV example:
#   mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGO_URI = os.environ.get("MONGO_URI", "")
MONGO_DB = "chatbot"
MONGO_COLLECTION = "history"

OLLAMA_URL = "http://localhost:11434/api/generate"
# Default model. Override at runtime in the TUI with `/model <name>`,
# or at startup by exporting MODEL_NAME before launching Python
# (e.g. `$env:MODEL_NAME="qwen3:8b"` on PowerShell, then `python -m chatbot.cli`).
# MODEL_NAME = os.environ.get("MODEL_NAME", "gemma3:4b")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:8b")

MAX_MESSAGES = 20
SYSTEM_PROMPT = "You are a helpful assistant."

NUM_GPU = 99
NUM_CTX = 4096
TEMPERATURE = 0.7
REPEAT_PENALTY = 1.1
