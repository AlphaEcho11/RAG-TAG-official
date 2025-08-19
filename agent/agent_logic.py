# rag_tag_agent.py
import os, datetime, json, time, uuid, shutil, sys, subprocess
import logging
import config

# --- Setup logging at the top of the file ---
# This configures a logger that writes to a file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="agent_activity.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser, Document
from .core_module import analyze_context

# --- Setup ---
persist_directory = "./chroma_db"
collection_name = "rag_tag_layered_memory_final"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="gemma-3-1b-it-qat", streaming=True)
def initialize_vectorstore(persist_dir: str, collection_name: str):
    """
    Initializes and returns the Chroma vector store instance.
    """
    return Chroma(
        collection_name="rag_tag_layered_memory_final",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./chroma_db"
    )

# --- Memory and Logging ---
def log_interaction_to_journal(query, response, source):
    # (This function remains unchanged)
    file_path = "cognitive_journal.json"; data = []; memory_entry = {"timestamp": datetime.datetime.now().isoformat(),"query": query,"response": response,"source": source}
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            try: data = json.load(f)
            except json.JSONDecodeError: data = []
    data.append(memory_entry)
    with open(file_path, 'w') as f: json.dump(data, f, indent=4)

# --- LAYER 3: New helper function for the feedback loop ---
def update_access_metadata(vectorstore, doc_id):
    """Updates the access count and timestamp for a retrieved document."""
    try:
        existing_doc = vectorstore.get(ids=[doc_id], include=["metadatas", "documents"])
        if not existing_doc or not existing_doc.get('ids'): return

        metadata = existing_doc['metadatas'][0]
        metadata['access_count'] = metadata.get('access_count', 0) + 1
        metadata['last_accessed_timestamp'] = int(time.time())
        
        # Overwrite the document with the same ID and updated metadata
        vectorstore.add_documents([Document(page_content=existing_doc['documents'][0], metadata=metadata)], ids=[doc_id])
        logger.info(f"--- [Feedback Loop] Updated metadata for memory {doc_id} ---")
    except Exception as e:
        logger.error(f"[Feedback Loop] Could not update metadata for {doc_id}: {e}")

def write_novel_thought_to_vector_memory(vectorstore, query, response, scope):
    """Writes a new thought with full metadata."""
    content = f"Question: {query}\nResponse: {response}"; doc_id = str(uuid.uuid4())
    doc = Document(page_content=content, metadata={
        "doc_id": doc_id, "scope": scope, "creation_timestamp": int(time.time()),
        "last_accessed_timestamp": int(time.time()), "access_count": 0, "utility_score": 0.0, "source": "novel_thought"
    })
    vectorstore.add_documents([doc], ids=[doc_id])
    logger.info(f"--- Novel Thought Written to Vector Memory in scope '{scope}'! ---")

# --- Core Agent Logic (Enhanced) ---
def process_query(vectorstore, question, conversation_history, current_scope="general", llm_model_name="gemma-3-1b-it-qat"):
    """
    Analyzes context, applies filtering, streams response, and logs.
    """
    # --- NEW: Initialize LLM based on the selected model ---
    model_details = config.AVAILABLE_MODELS.get(llm_model_name, config.AVAILABLE_MODELS[config.DEFAULT_MODEL])
    llm = ChatOpenAI(
        base_url=model_details["base_url"], 
        api_key=model_details["api_key"], 
        model=llm_model_name, 
        streaming=True
    )

    # --- Step 1: Call the Enhanced CORE Module ---
    core_analysis = analyze_context(question, conversation_history)
    
    focused_query = question
    if core_analysis:
        relevance_hierarchy = core_analysis['focused_execution']['relevance_hierarchy']
        focused_query = f"{question} - focusing on: {', '.join(relevance_hierarchy)}"
        logger.info(f"--- Focused Query based on CORE: {focused_query} ---")


    # --- Step 2: Perform Layer 1 Retrieval Logic (Unchanged) ---
    existing_docs = vectorstore.get(include=["metadatas"])
    all_known_scopes = {md['scope'] for md in existing_docs['metadatas'] if 'scope' in md}
    scopes_to_query = {current_scope}
    if "::" in current_scope: scopes_to_query.add(current_scope.split("::")[0])
    for known_scope in all_known_scopes:
        if known_scope.startswith(f"{current_scope}::"): scopes_to_query.add(known_scope)
    
    where_filter = {"scope": {"$in": list(scopes_to_query)}}
    retrieved_docs = vectorstore.similarity_search_with_score(focused_query, k=1, filter=where_filter)
    
    full_response = ""
    source = ""
    
    # --- Step 3: Generate Response, Now with Full CORE Integration ---
    if retrieved_docs and retrieved_docs[0][1] < 1.0:
        source = "RAG_protocol_log"
        retrieved_doc = retrieved_docs[0][0]
        logger.info(f"Activating RAG Protocol in scopes {list(scopes_to_query)}. Relevant memory found.")
        
        retrieved_id = retrieved_doc.metadata.get('doc_id')
        if retrieved_id: update_access_metadata(vectorstore, retrieved_id)

        adaptive_goal = core_analysis['situational_awareness']['adaptive_goal'] if core_analysis else "Answer the user's question based on the provided context."
        
        # --- NEW FEATURE: Check if we need to explain any limitations ---
        limitation_text = ""
        if core_analysis and core_analysis['focused_execution']['articulate_limitations']:
            constraints = core_analysis['situational_awareness']['operational_constraints']
            limitation_text = f"As an AI with specific operational boundaries, I've adapted your request to provide the most helpful information possible. Based on my analysis, here is information regarding: {constraints}.\n\n"
        
        final_prompt = f"{limitation_text}Context: {retrieved_doc.page_content}\n\nBased on this context, your goal is to: **{adaptive_goal}**. Fulfill this goal by answering the user's question: {question}\nAnswer:"
        chain = llm | StrOutputParser()
        for chunk in chain.stream(final_prompt):
            yield chunk; full_response += chunk
        
    else:
        # (This part for Novel Thought remains largely the same)
        source = "novel_thought_log"
        logger.info(f"Activating Novel Thought Protocol in scope '{current_scope}'. No relevant memory found.")
        chain = ChatPromptTemplate.from_template("{question}") | llm | StrOutputParser()
        for chunk in chain.stream({"question": question}):
            yield chunk; full_response += chunk

    logger.info(f"Full Response Generated by Agent (Novel Thought): {full_response[:100]}...") # Log a snippet
    
    if source == "RAG_protocol_log":
        log_interaction_to_journal(question, "Used RAG to generate response.", source)
    elif source == "novel_thought_log":
        log_interaction_to_journal(question, full_response, source)
        write_novel_thought_to_vector_memory(vectorstore, question, full_response, current_scope)
        
# --- Integrated Test Block ---
if __name__ == '__main__':
    print("--- Testing Full RAG-TAG System (Layer 1, 2, & 3) ---")
    if os.path.exists(persist_directory): shutil.rmtree(persist_directory); print("Cleared old database for a fresh test.")
    
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_function, persist_directory=persist_directory)

    # --- LAYER 3: Start the Huey consumer as an integrated background process ---
    print("\n--- Starting Integrated Huey Consumer ---")
    # This command finds your Python executable and uses it to run the Huey consumer module
    consumer_command = [sys.executable, '-m', 'huey.bin.huey_consumer', 'tasks.huey']
    # Popen starts the process without blocking the rest of our script
    consumer_process = subprocess.Popen(consumer_command)
    print("--- Huey Consumer is running in the background. ---")
    
    conversation_history = []
    test_scope = "project_alpha::auth_feature"
    
    try:
        print("\n\n--- TEST STEP 1: Creating a memory ---")
        question1 = "What is the standard expiration time for a user session token?"
        full_response1 = "".join(list(process_query(vectorstore, question1, conversation_history, current_scope=test_scope)))
        conversation_history.append({"query": question1, "response": full_response1})

        print("\n\n--- TEST STEP 2: Accessing the memory to trigger feedback loop ---")
        question2 = "Why do our session tokens expire?"
        full_response2 = "".join(list(process_query(vectorstore, question2, conversation_history, current_scope=test_scope)))
        conversation_history.append({"query": question2, "response": full_response2})

        print("\n\n--- WAITING for Huey to run the curator task (approx. 1 min)... ---")
        time.sleep(75) # Wait long enough for the 1-minute scheduled task to fire

    finally:
        # --- Clean up the background process when the test is done ---
        print("\n--- Shutting down Integrated Huey Consumer ---")
        consumer_process.terminate()
        consumer_process.wait()
        print("--- Test complete. ---")