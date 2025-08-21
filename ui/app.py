# ui/app.py
import gradio as gr
import os
import json
from agent import omni_loader
from agent.agent_logic import initialize_vectorstore, process_query, write_novel_thought_to_vector_memory, generate_image_with_tool
import config

# (initialize_vectorstore and app_state remain the same)
vectorstore = initialize_vectorstore(
    persist_dir=config.PERSIST_DIRECTORY,
    collection_name="rag_tag_layered_memory_final"
)
app_state = {
    "llm_model": config.DEFAULT_MODEL
}


def update_journal_display():
    """Reads and formats the cognitive journal for display."""
    try:
        with open(config.COGNITIVE_JOURNAL, "r") as f:
            return json.dumps(json.load(f), indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.dumps({"status": "Journal is empty..."}, indent=2)

# --- NEW: Refactored Multimodal Chat Interface ---
def multimodal_chat_interface(text_message, file_upload, history):
    # ... (file upload logic is the same) ...
    
    elif text_message:
        history.append((text_message, ""))
        
        response_generator = process_query(vectorstore, text_message, [], "general", app_state["llm_model"])
        
        full_text_response = ""
        for result in response_generator:
            if isinstance(result, dict) and result.get("type") == "image":
                history.append((None, (result["path"],)))
            
            # --- NEW: Handle the UI message for file requests ---
            elif isinstance(result, dict) and result.get("type") == "ui_message":
                # The agent is asking for a file, so we update the last message
                history[-1] = (text_message, result.get("content"))
            
            else:
                # Otherwise, it's a standard text chunk
                full_text_response += str(result)
                history[-1] = (text_message, full_text_response)    history = history or []
    
    if file_upload:
        # If a file is present, delegate EVERYTHING to the Omni-Loader
        print(f"--- [UI] File detected. Sending '{file_upload.name}' to Omni-Loader... ---")
        history.append((f"Processing file: **{os.path.basename(file_upload.name)}** with query: \"{text_message}\"", None))
        
        llm_config = config.AVAILABLE_MODELS[app_state["llm_model"]]
        
        # The Omni-Loader now handles everything: processing, memory, and generating a user-facing response.
        response_content = omni_loader.process_input(file_upload.name, text_message, vectorstore, llm_config)
        
        history[-1] = (history[-1][0], response_content) # Update the placeholder message with the actual response

    elif text_message:
        # Text-only logic remains the same, driven by the CORE module
        history.append((text_message, ""))
        response_generator = process_query(vectorstore, text_message, [], "general", app_state["llm_model"])
        
        full_text_response = ""
        for result in response_generator:
            if isinstance(result, dict) and result.get("type") == "image":
                history.append((None, (result["path"],)))
            else:
                full_text_response += str(result) # Ensure chunk is a string
                history[-1] = (text_message, full_text_response)

    # Clear the input fields after processing
    return history, gr.update(value=""), gr.update(value=None)


# --- IMPORTANT: Update the event listeners to handle the new return signature ---
msg_textbox.submit(
    multimodal_chat_interface,
    [msg_textbox, upload_button, chatbot],
    [chatbot, msg_textbox, upload_button] # We now also clear the upload button
)
upload_button.upload(
    multimodal_chat_interface,
    [msg_textbox, upload_button, chatbot],
    [chatbot, msg_textbox, upload_button] # And here too    history = history or []
    llm_config = config.AVAILABLE_MODELS[app_state["llm_model"]]

    if file_upload:
        # This part remains the same: file uploads go to the omni_loader
        history.append(((file_upload.name,), None))
        response_content = process_input(file_upload.name, text_message, vectorstore, llm_config)
        history.append((text_message, response_content))
    
    elif text_message:
        # This part is now much simpler
        history.append((text_message, ""))
        
        response_generator = process_query(vectorstore, text_message, [], "general", app_state["llm_model"])
        
        full_text_response = ""
        for result in response_generator:
            if isinstance(result, dict) and result.get("type") == "image":
                # If we get an image dictionary, display the image
                history.append((None, (result["path"],)))
            else:
                # Otherwise, it's a text chunk from the RAG pipeline
                full_text_response += result
                history[-1] = (text_message, full_text_response) # Update the last entry

    return history, "", None    history = history or []
    llm_config = config.AVAILABLE_MODELS[app_state["llm_model"]]

    if file_upload:
        history.append(((file_upload.name,), None))
        # Now we pass the vectorstore to process_input
        response_content = process_input(file_upload.name, text_message, vectorstore, llm_config)
        history.append((text_message, response_content))
    else:
        if "generate an image of" in text_message.lower() or "create a picture of" in text_message.lower():
            prompt = text_message.replace("generate an image of", "").strip()
            
            # --- Step 1: Generate the image ---
            image_path = generate_image_with_tool(prompt)
            history.append((text_message, None))
            history.append((None, (image_path,)))
            
            # --- Step 2: Create and write the memory packet ---
            memory_content = f"The user asked me to generate an image with the prompt: '{prompt}'. I created the image and saved it at {os.path.basename(image_path)}."
            print("--- [Generator] Writing image creation to vector memory... ---")
            write_novel_thought_to_vector_memory(
                vectorstore,
                query=f"Memory of generating image for prompt: {prompt}",
                response=memory_content,
                scope="generation_log" # A dedicated scope for created content
            )
        else:
            # --- Regular RAG text processing ---
            response_generator = process_text_with_rag(vectorstore, text_message, [], "general", app_state["llm_model"])
            full_response = "".join(list(response_generator))
            history.append((text_message, full_response))

    return history, "", None    """Handles both text and file uploads."""
    history = history or []
    response_text = ""

    if file_upload:
        # If a file is uploaded, process it with the Omni-Loader
        history.append(((file_upload.name,), None)) # Display the uploaded file
        # We need a placeholder for the llm object for now
        llm_placeholder = {} 
        response_text = process_input(file_upload.name, text_message, vectorstore, llm_placeholder)
    else:
        # If it's just text, we'll eventually route this to the text-only part of the agent
        # For now, a simple echo
        from agent.agent_logic import process_query # Import here to avoid circularity if needed
        
        # This part needs to be wired up to the actual text processing logic
        # For now, let's just confirm it works
        response_generator = process_query(vectorstore, text_message, [], "general", app_state["llm_model"])
        response_text = "".join(list(response_generator))

    history.append((text_message, response_text))
    return history, ""

# --- MODIFIED: update_model now returns a status update ---
def update_model(new_model):
    """Updates the session's LLM model and provides user feedback."""
    app_state["llm_model"] = new_model
    # This will create a temporary popup notification in the Gradio UI
    return gr.Info(f"Switched session LLM to: {new_model}")

# --- GRADIO UI (Now with Multimodal Capabilities) ---
with gr.Blocks(theme='default') as demo:
    gr.Markdown("# RAG-TAG Agent Console (Omnimodal)")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=600, bubble_full_width=False)
            
            with gr.Row():
                msg_textbox = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Enter text and press enter, or upload a file",
                    container=False
                )
                upload_button = gr.UploadButton("📁", file_types=["image", ".pdf", ".txt", ".docx"])

        with gr.Column(scale=1):
            # (System Config and Journal remain the same)
            with gr.Accordion("System Configuration", open=True):
                model_dropdown = gr.Dropdown(
                    label="LLM Model (Session)",
                    choices=list(config.AVAILABLE_MODELS.keys()),
                    value=app_state["llm_model"]
                )
            with gr.Accordion("Cognitive Journal", open=False):
                journal_output = gr.Code(
                    label="Live Feed", language="json", interactive=False
                )
    status_update = gr.Textbox(label="Status", visible=False) # A hidden component to receive the update                
                
    # --- Event Listeners ---
    msg_textbox.submit(
        multimodal_chat_interface,
        [msg_textbox, upload_button, chatbot],
        [chatbot, msg_textbox]
    )
    upload_button.upload(
        multimodal_chat_interface,
        [msg_textbox, upload_button, chatbot],
        [chatbot, msg_textbox]
    )
    
    chatbot.change(fn=update_journal_display, inputs=None, outputs=journal_output)
    model_dropdown.change(update_model, inputs=model_dropdown, outputs=status_update)

if __name__ == "__main__":
    demo.launch(pwa=True)