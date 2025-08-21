# ui/app.py
import gradio as gr
import os
import json
import mimetypes
import config
from agent import omni_loader
from agent.agent_logic import initialize_vectorstore, process_query, log_interaction_to_journal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# --- INITIALIZATION ---
vectorstore = initialize_vectorstore(
    persist_dir=config.PERSIST_DIRECTORY,
    collection_name="rag_tag_layered_memory_final"
)
app_state = {
    "llm_model": config.DEFAULT_MODEL
}

# --- UI HELPER FUNCTIONS ---
def update_journal_display():
    try:
        with open(config.COGNITIVE_JOURNAL, "r") as f:
            all_entries = json.load(f)
            return json.dumps(all_entries[-3:], indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.dumps({"status": "Journal is empty..."}, indent=2)

def update_model(new_model):
    app_state["llm_model"] = new_model
    return gr.Info(f"Switched session LLM to: {new_model}")

# --- CORE CHAT INTERFACE LOGIC (REFACTORED FOR 'MESSAGES' FORMAT) ---
def multimodal_chat_interface(text_message, files, history):
    new_history = list(history) if history else []
    
    # Create the user's turn as a single message with multiple content parts
    user_content = []
    if text_message:
        user_content.append({"type": "text", "text": text_message})
    if files:
        for file_obj in files:
            new_history.append({"role": "user", "content": (file_obj.name,)})

    if text_message:
        new_history.append({"role": "user", "content": text_message})

    # Only proceed if there is some user input
    if text_message or files:
        new_history.append({"role": "assistant", "content": "Thinking..."})

        # Process in the background
        llm_config = config.AVAILABLE_MODELS[app_state["llm_model"]]
        timeout = llm_config.get("request_timeout")
        if llm_config["model_provider"] == "google":
            llm = ChatGoogleGenerativeAI(model=llm_config["model_name"], google_api_key=llm_config["api_key"])
        else: # Add other providers here as needed
            llm = ChatOpenAI(model=llm_config["model_name"], base_url=llm_config.get("base_url"), api_key=llm_config.get("api_key"))

        final_response = ""
        log_query = text_message
        source = "user_prompt"
        
        image_files = [f for f in files if mimetypes.guess_type(f.name)[0] and mimetypes.guess_type(f.name)[0].startswith("image/")] if files else []
        if image_files:
            image_paths = [img.name for img in image_files]
            log_query += f" (with {len(image_paths)} image(s))"
            final_response = omni_loader.handle_vision_input(image_paths, text_message, vectorstore, llm_config, llm)
            source = "multi_image_upload"

        doc_files = [f for f in files if f not in image_files] if files else []
        if doc_files:
            processed_docs = [os.path.basename(doc.name) for doc in doc_files]
            log_query += f" (with {len(processed_docs)} document(s))"
            for doc in doc_files:
                omni_loader.handle_document_input(doc.name, vectorstore)
            final_response = f"I have successfully memorized {len(processed_docs)} document(s).\n\nWhat would you like to know about them?"
            source = "multi_document_upload"

        if not image_files and not doc_files and text_message:
            # Handle text-only streaming response
            response_generator = process_query(vectorstore, text_message, [], "general", app_state["llm_model"])
            full_text_response = ""
            for chunk in response_generator:
                full_text_response += str(chunk)
                new_history[-1]["content"] = full_text_response
            final_response = full_text_response
            source = "text_rag"
        else:
            new_history[-1]["content"] = final_response

        log_interaction_to_journal(query=log_query, response=final_response, source=source)

    return new_history, gr.update(value=""), gr.update(value=None)

# --- GRADIO UI LAYOUT ---
with gr.Blocks(theme='default') as demo:
    gr.Markdown("# RAG-TAG Agent Console (Omnimodal)")
    with gr.Row():
        with gr.Column(scale=2):
            # --- THIS IS THE FIX for the warning ---
            chatbot = gr.Chatbot(
                label="Chat",
                height=600,
                bubble_full_width=False,
                render_markdown=True,
                type="messages" # Set the type to 'messages'
            )
            file_uploader = gr.File(label="Upload Files", file_count="multiple", type="filepath")
            with gr.Row():
                msg_textbox = gr.Textbox(scale=4, show_label=False, placeholder="Enter text to accompany files or send a message", container=False)
                send_button = gr.Button("Send", variant="primary", scale=1)
        with gr.Column(scale=1):
            with gr.Accordion("System Configuration", open=True):
                model_dropdown = gr.Dropdown(label="LLM Model (Session)", choices=list(config.AVAILABLE_MODELS.keys()), value=app_state["llm_model"])
            with gr.Accordion("Cognitive Journal (Last 3 Entries)", open=False):
                journal_output = gr.Code(label="Live Feed", language="json", interactive=False)
            status_update = gr.Textbox(label="Status", visible=False)

    # --- EVENT LISTENERS ---
    outputs_to_update = [chatbot, msg_textbox, file_uploader]

    send_button.click(multimodal_chat_interface, [msg_textbox, file_uploader, chatbot], outputs_to_update)
    msg_textbox.submit(multimodal_chat_interface, [msg_textbox, file_uploader, chatbot], outputs_to_update)
    
    chatbot.change(fn=update_journal_display, inputs=None, outputs=journal_output)
    model_dropdown.change(update_model, inputs=model_dropdown, outputs=status_update)

if __name__ == "__main__":
    demo.launch(pwa=True)