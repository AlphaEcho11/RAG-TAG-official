# ui/app.py
import gradio as gr
import os
import json
from agent.agent_logic import initialize_vectorstore, process_query
import config

# --- REMOVED ---
# The watchdog, threading, and time imports are no longer needed.
# The JournalHandler class, start_watcher function, and threading logic have been removed.

# --- SIMPLIFIED JOURNAL READER ---
# This is a much simpler function that just reads and returns the journal's content.
def update_journal_display():
    """Reads and formats the cognitive journal for display."""
    try:
        with open(config.COGNITIVE_JOURNAL, "r") as f:
            # Read the file and format for display
            return json.dumps(json.load(f), indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return a waiting message if the file doesn't exist or is empty
        return json.dumps({"status": "Journal is empty or not yet created..."}, indent=2)

# --- INITIALIZATION ---
vectorstore = initialize_vectorstore(
    persist_dir=config.PERSIST_DIRECTORY,
    collection_name="rag_tag_layered_memory_final"
)
app_state = {
    "llm_model": config.DEFAULT_MODEL
}

def chat_interface(message, history):
    """Handles user interaction and streams the agent's response."""
    history.append({"role": "user", "content": message})
    conversation_history = []
    if history:
        for i, message_pair in enumerate(history):
            if message_pair["role"] == "user":
                assistant_response = history[i+1]["content"] if i+1 < len(history) and history[i+1]["role"] == "assistant" else ""
                conversation_history.append({"query": message_pair["content"], "response": assistant_response})
    
    current_scope = "project_alpha"
    
    response_generator = process_query(
        vectorstore,
        message,
        conversation_history,
        current_scope,
        llm_model_name=app_state["llm_model"]
    )
    
    full_response = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in response_generator:
        full_response += chunk
        history[-1]["content"] = full_response
        yield history, ""

def update_model(new_model):
    """Updates the session's LLM model."""
    app_state["llm_model"] = new_model
    gr.Info(f"Switched session LLM to: {new_model}")

# --- GRADIO UI ---
with gr.Blocks(theme='default') as demo:
    gr.Markdown("# RAG-TAG Agent Console")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type='messages', label="Chat", height=600)
            msg = gr.Textbox(label="Query", placeholder="Type your message here...")
            clear = gr.Button("Clear")

        with gr.Column(scale=1):
            with gr.Accordion("System Configuration", open=True):
                model_dropdown = gr.Dropdown(
                    label="LLM Model (Session)",
                    choices=list(config.AVAILABLE_MODELS.keys()),
                    value=app_state["llm_model"]
                )
                gr.Markdown(f"**Database Path:** `{config.PERSIST_DIRECTORY}`")
                gr.Markdown(f"**Journal Path:** `{config.COGNITIVE_JOURNAL}`")
            
            with gr.Accordion("Cognitive Journal", open=False):
                journal_output = gr.Code(
                    label="Live Feed", language="json", interactive=False
                )

    # --- Event Listeners ---
    msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)
    model_dropdown.change(update_model, inputs=model_dropdown)

    # --- RELIABLE UPDATE: Refresh journal after chatbot responds ---
    chatbot.change(fn=update_journal_display, inputs=None, outputs=journal_output)

if __name__ == "__main__":
    demo.launch()