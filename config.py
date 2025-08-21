# config.py
import os
from dotenv import load_dotenv

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
        "model_name": "gemma-3-4B-it-qat",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
    # 2. Google Gemini for Cloud Omnimodal
    "google-gemini-2.5-pro": {
        "model_provider": "google",
        "model_name": "gemini-2.5-pro",
        "api_key": os.getenv("GOOGLE_API_KEY"),
    },
    # 3. OpenAI
    "openai-gpt-4o": {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    # 4. Anthropic
    "anthropic-claude-3-sonnet": {
        "model_provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    }
}
# Set the new local model as the default
DEFAULT_MODEL = "lm-studio-gemma-3-4b-it"

# --- Embedding Settings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"