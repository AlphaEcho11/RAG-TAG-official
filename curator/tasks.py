# /curator/tasks.py
from huey import SqliteHuey, crontab

# --- Huey Configuration ---

# Current setup for local/lightweight production. Simple and effective.
huey = SqliteHuey(filename='curator_tasks.db')

# --- FOR HEAVY PRODUCTION ENVIRONMENTS ---
# To scale up, you would switch to a Redis backend. This requires a running
# Redis server and the 'redis' Python package (`pip install redis`).
# from huey import RedisHuey
# huey = RedisHuey('rag-tag-curator', host='localhost', port=6379)
# -------------------------------------------

# --- Imports for initializing the curator's tools ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- The Key Change: Relative Import ---
# This now correctly imports from curator_logic.py within the same folder.
from .curator_logic import prune_memories, synthesize_memories

# 1. Configure Huey to use a simple SQLite file. No Redis needed.
huey = SqliteHuey(filename='curator_tasks.db')

# 2. Define a scheduled task for pruning
# Set to run every minute for easy testing. Change for production.
@huey.periodic_task(crontab(minute='*/1'))
def run_periodic_pruning():
    """
    Huey task to run the memory pruning function.
    """
    print("--- [Huey] Kicking off periodic pruning task. ---")
    # The worker initializes its own connection to the tools it needs
    vectorstore = Chroma(
        collection_name="rag_tag_layered_memory_final",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./chroma_db"
    )
    prune_memories(vectorstore)

# 3. Define a placeholder for the synthesis task
@huey.periodic_task(crontab(hour=2, day_of_week=0)) # 2 AM on Sunday
def run_weekly_synthesis():
    """
    Huey task to run the memory synthesis function.
    """
    print("--- [Huey] Kicking off weekly synthesis task. ---")
    # This task would also initialize its own vectorstore and llm
    # vectorstore = Chroma(...)
    # llm = ChatOpenAI(...)
    # synthesize_memories(vectorstore, llm)