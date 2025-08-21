# agent_logic.py
import os
import datetime
import json
import time
import uuid
import logging
import config

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, Document
from langchain.retrievers.multi_query import MultiQueryRetriever

from .core_module import analyze_context

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="agent_activity.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def initialize_vectorstore(persist_dir: str, collection_name: str):
    """Initializes and returns the Chroma vector store instance."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL),
        persist_directory=persist_dir
    )

# --- ADD THIS FUNCTION BACK ---
def log_interaction_to_journal(query, response, source):
    """Appends a record of an interaction to the cognitive journal file."""
    file_path = config.COGNITIVE_JOURNAL
    data = []
    memory_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "response": response,
        "source": source
    }
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(memory_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def write_novel_thought_to_vector_memory(vectorstore, query, response, scope):
    """Writes a new thought with full metadata to the vector store."""
    content = f"Query: {query}\nResponse: {response}"
    doc_id = str(uuid.uuid4())
    doc = Document(page_content=content, metadata={
        "doc_id": doc_id, "scope": scope, "creation_timestamp": int(time.time()),
        "last_accessed_timestamp": int(time.time()), "access_count": 0
    })
    vectorstore.add_documents([doc], ids=[doc_id])
    logger.info(f"--- Novel Thought written to Vector Memory in scope '{scope}' ---")

def generate_image_with_tool(prompt: str) -> str:
    """A 'tool' the agent can use to generate an image."""
    print(f"--- [Generator Tool] Received prompt: '{prompt}' ---")
    placeholder_image_path = "ui/placeholder.png"
    if not os.path.exists(placeholder_image_path):
        from PIL import Image
        img = Image.new('RGB', (400, 200), color = 'gray')
        # Ensure the ui directory exists
        os.makedirs(os.path.dirname(placeholder_image_path), exist_ok=True)
        img.save(placeholder_image_path)
    return placeholder_image_path

def process_text_with_rag(vectorstore, question, conversation_history, current_scope, llm):
    """
    Handles the RAG pipeline using an advanced Multi-Query Retriever to ensure
    document memories are always accessible.
    """
    # --- DIAGNOSTIC PRINT ---
    print("\n--- [RAG] Verifying RAG Function: ADVANCED RETRIEVER v2 ---")
    
    # 1. Define the scopes to search. This is the critical step.
    scopes_to_search = {current_scope, "document_ingest"}
    search_filter = {"scope": {"$in": list(scopes_to_search)}}
    
    # --- DIAGNOSTIC PRINT ---
    print(f"--- [RAG] Searching for memories in scopes: {list(scopes_to_search)} ---")
    
    # 2. Set up the advanced retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={'filter': search_filter})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )

    # 3. Invoke the retriever
    retrieved_docs = multi_query_retriever.invoke(question)
    
    # 4. Process and respond
    if not retrieved_docs:
        context = "No relevant information was found in my memory, even after generating multiple queries."
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # --- DIAGNOSTIC PRINT ---
    print(f"--- [RAG] Found {len(retrieved_docs)} relevant documents to answer the question. ---")
    
    prompt = ChatPromptTemplate.from_template("Based on the following context, answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    chain = prompt | llm | StrOutputParser()
    
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk    
        """
    Handles the RAG pipeline for text-based queries using a more advanced
    Multi-Query Retriever to improve accuracy.
    """
    print(f"--- [RAG] Activating Multi-Query Retriever for question: '{question}' ---")
    
    # 1. Set up the advanced retriever
    scopes_to_search = {current_scope, "document_ingest"}
    search_filter = {"scope": {"$in": list(scopes_to_search)}}
    
    base_retriever = vectorstore.as_retriever(search_kwargs={'filter': search_filter})
    
    # The MultiQueryRetriever uses the LLM to generate better search queries
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )

    # 2. Invoke the retriever to get relevant documents
    retrieved_docs = multi_query_retriever.invoke(question)
    
    # 3. Process and respond
    if not retrieved_docs:
        context = "No relevant information was found in my memory, even after generating multiple queries."
    else:
        # Join the content of all found documents to form the context
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    print(f"--- [RAG] Found {len(retrieved_docs)} relevant documents. ---")
    
    prompt = ChatPromptTemplate.from_template("Based on the following context, answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    chain = prompt | llm | StrOutputParser()
    
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk

# --- The CORE Orchestrator ---
def process_query(vectorstore, question, conversation_history, current_scope="general", llm_model_name=config.DEFAULT_MODEL):
    """
    This is the main entry point. It initializes the LLM, gets a plan from the
    CORE module, and executes the recommended tool.
    """
    try:
        model_config = config.AVAILABLE_MODELS.get(llm_model_name, config.AVAILABLE_MODELS[config.DEFAULT_MODEL])
        
        if model_config["model_provider"] == "openai":
            llm = ChatOpenAI(model=model_config["model_name"], api_key=model_config["api_key"])
        elif model_config["model_provider"] == "google":
            llm = ChatGoogleGenerativeAI(model=model_config["model_name"], google_api_key=model_config["api_key"])
        elif model_config["model_provider"] == "anthropic":
            llm = ChatAnthropic(model=model_config["model_name"], api_key=model_config["api_key"])
        else: # Default to local
            llm = ChatOpenAI(model=model_config["model_name"], base_url=model_config.get("base_url"), api_key=model_config.get("api_key"))

        core_plan = analyze_context(question, conversation_history, llm)

        if not core_plan or 'recommended_action' not in core_plan:
            print("--- [Agent Logic] CORE analysis failed. Defaulting to text_rag. ---")
            yield from process_text_with_rag(vectorstore, question, conversation_history, current_scope, llm)
            return

        action = core_plan['recommended_action']
        tool_to_use = action.get("tool")
        parameters = action.get("parameters", {})

        print(f"--- [Agent Logic] CORE Plan received. Executing tool: '{tool_to_use}' ---")

        if tool_to_use == 'image_generator':
            prompt = parameters.get('prompt', question)
            image_path = generate_image_with_tool(prompt)
            yield {"type": "image", "path": image_path}
            memory_content = f"Following a CORE plan, I generated an image based on the prompt: '{prompt}'."
            write_novel_thought_to_vector_memory(vectorstore, f"Memory of generating image: {prompt}", memory_content, "generation_log")
        elif tool_to_use == 'request_file':
            request_message = parameters.get('request_message', "To proceed, please upload the relevant file.")
            yield {"type": "ui_message", "content": request_message}
        else: # Default to text_rag
            yield from process_text_with_rag(vectorstore, question, conversation_history, current_scope, llm)

    except Exception as e:
        error_message = str(e)
        print(f"--- [Agent Logic] CRITICAL ERROR: {error_message} ---")
        
        if "api_key" in error_message.lower():
            user_facing_error = f"**API Error:** Could not connect to the model. Please check if your API key for **{llm_model_name}** is correct and has sufficient credits."
        else:
            user_facing_error = f"**System Error:** An unexpected issue occurred while processing your request. Please check the console logs for details."
            
        yield {"type": "ui_message", "content": user_facing_error}