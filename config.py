# config.py
import os
from dotenv import load_dotenv

# --- Feature Flags ---
# Set to False to run the agent in a lightweight, text-only mode.
OMNIMODAL_FEATURES_ENABLED = True

# --- Core Paths ---

# Load environment variables from a .env file
load_dotenv()

# --- Core Paths ---
PERSIST_DIRECTORY = "./chroma_db_prod"
COGNITIVE_JOURNAL = "cognitive_journal.json"

# --- LLM Settings ---
AVAILABLE_MODELS = {
    # 1. Local Multimodal Model
    "lm-studio-gemma-3-4b-it": {
        "model_provider": "local",
        "model_name": "lmstudio-community/gemma-3-4b-it-gguf", # Use the server identifier, not the filename
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "request_timeout": 180,  # Timeout in seconds
    },
    # 2. Google Gemini for Cloud Omnimodal
    "google-gemini-2.5-pro": {
        "model_provider": "google",
        "model_name": "gemini-2.5-pro",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "request_timeout": 180,  # Timeout in seconds
    },
    # 3. OpenAI
    "openai-gpt-4o": {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "request_timeout": 180,  # Timeout in seconds
    },
    # 4. Anthropic
    "anthropic-claude-3-sonnet": {
        "model_provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "request_timeout": 180,  # Timeout in seconds
    }
}
# Set the new local model as the default
DEFAULT_MODEL = "lm-studio-gemma-3-4b-it"

# --- Embedding Settings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"