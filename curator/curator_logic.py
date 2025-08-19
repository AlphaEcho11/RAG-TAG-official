# curator_logic.py
import time
import logging

# --- Setup logging at the top of the file ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="curator_activity.log", # Give it a separate log file
    filemode="a"
)
logger = logging.getLogger(__name__)

def prune_memories(vectorstore, max_age_days=30, min_access_count=2):
    """
    Finds and deletes memories that are old and have not been accessed frequently.
    """
    logger.info("--- [Curator] Starting pruning process ---")
    all_memories = vectorstore.get(include=["metadatas"])
    if not all_memories or not all_memories.get('ids'):
        logger.info("--- [Curator] No memories found to prune. ---")
        return

    ids_to_delete = []
    current_time = int(time.time())
    max_age_seconds = max_age_days * 24 * 60 * 60

    for i, metadata in enumerate(all_memories['metadatas']):
        doc_id = all_memories['ids'][i]
        access_count = metadata.get('access_count', 0)
        creation_time = metadata.get('creation_timestamp', 0)
        
        # Prune if it's older than max_age_days AND accessed less than min_access_count
        if (current_time - creation_time) > max_age_seconds and access_count < min_access_count:
            ids_to_delete.append(doc_id)
            
    if ids_to_delete:
        logger.info(f"--- [Curator] Pruning {len(ids_to_delete)} stale memories. ---")
        vectorstore.delete(ids=ids_to_delete)
    else:
        logger.info("--- [Curator] No memories to prune. ---")

def synthesize_memories(vectorstore, llm):
    """
    (Conceptual) Finds clusters of related memories and synthesizes them.
    """
    logger.info("\n--- [Curator] Starting synthesis process (Conceptual) ---")
    # This is where you would implement a more complex data science workflow:
    # 1. Get all memories with high access counts.
    # 2. Use a clustering algorithm on their embeddings to find related groups.
    # 3. For each group, use the LLM to create a single, dense summary.
    # 4. Add the new summary memory and (optionally) delete the originals.
    logger.info("--- [Curator] Synthesis complete. ---")