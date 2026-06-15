"""Runtime constants for the local LLM chatbot."""

HOST = "127.0.0.1"
PORT = 9000

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

MAX_MESSAGES = 20
SYSTEM_PROMPT = "You are a helpful assistant."

NUM_GPU = 99
NUM_CTX = 4096
TEMPERATURE = 0.7
REPEAT_PENALTY = 1.1
