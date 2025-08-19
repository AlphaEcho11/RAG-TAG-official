# config.py

# --- Core Paths ---
PERSIST_DIRECTORY = "./chroma_db_prod"
COGNITIVE_JOURNAL = "cognitive_journal.json"

# --- LLM Settings ---
# A dictionary of available models
AVAILABLE_MODELS = {
    "gemma-3-1b-it-qat": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
    "google/gemma-3n-e4b": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    }
}
# The default model to use on startup
DEFAULT_MODEL = "gemma-3-1b-it-qat"

# --- Embedding Settings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"