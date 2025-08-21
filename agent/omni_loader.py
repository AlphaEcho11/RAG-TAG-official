# agent/omni_loader.py
import mimetypes
import os
import base64
import requests
import time
import uuid
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

from .agent_logic import write_novel_thought_to_vector_memory

def encode_image(image_path):
    """Encodes an image file into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def handle_vision_input(image_paths: list, text_query: str, vectorstore, llm_config, llm):
    """
    Handles single or multiple vision inputs by building a bundled prompt
    and using the correct, pre-initialized LLM client.
    """
    print(f"--- [Omni-Loader] Handling vision input for {len(image_paths)} image(s). ---")
    try:
        content_parts = [{"type": "text", "text": text_query}]
        image_basenames = [os.path.basename(p) for p in image_paths]
        for image_path in image_paths:
            base64_image = encode_image(image_path)
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        vision_response = ""
        if llm_config["model_provider"] == "google":
            print("--- [Omni-Loader] Using Google Gemini client for multi-image vision ---")
            msg = llm.invoke([HumanMessage(content=content_parts)])
            print("--- [Omni-Loader] Received response from Google API. ---")
            vision_response = msg.content
        else: # Local / OpenAI-compatible
            print("--- [Omni-Loader] Using local/OpenAI client for multi-image vision ---")
            headers = {"Content-Type": "application/json"}
            payload = {"model": llm_config.get("model_name"), "messages": [{"role": "user", "content": content_parts}]}
            response = requests.post(f"{llm_config['base_url']}/chat/completions", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            vision_response = response.json()['choices'][0]['message']['content']

        # Check if the API returned an empty response, which can indicate content filtering.
        if not vision_response or not vision_response.strip():
            print("--- [Omni-Loader] WARNING: The API returned an empty response. ---")
            return "**Error:** The remote API returned an empty response. This could be due to the content of the prompt or images triggering the provider's safety filters. Please try different images or a different query."

        memory_content = f"The user uploaded {len(image_paths)} images ({', '.join(image_basenames)}) with the query: '{text_query}'. My analysis was: {vision_response}"
        write_novel_thought_to_vector_memory(vectorstore, f"Memory of analyzing images: {', '.join(image_basenames)}", memory_content, "vision_log")
        return vision_response
    except Exception as e:
        error_msg = f"An unexpected error occurred while analyzing the image(s): {e}"
        print(f"--- [Omni-Loader] ERROR: {error_msg} ---")
        return f"**Error:** {error_msg}"

def handle_document_input(file_path, vectorstore):
    """
    Extracts, chunks, and memorizes text from a document for RAG.
    """
    print(f"--- [Omni-Loader] Starting document processing for: {os.path.basename(file_path)} ---")
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        full_text = extract_text_from_document(file_path, mime_type)
        if not full_text or not full_text.strip():
            return f"**Notice:** Could not extract readable text from **{os.path.basename(file_path)}**."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        
        metadatas = [{"source": os.path.basename(file_path), "scope": "document_ingest"} for _ in chunks]
        
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        print(f"--- [Omni-Loader] Wrote {len(chunks)} chunks to vector memory. ---")
        return f"Successfully read and memorized {len(chunks)} sections from **{os.path.basename(file_path)}**."
    except Exception as e:
        error_msg = f"An unexpected error occurred while processing the document. (Details: {e})"
        print(f"--- [Omni-Loader] ERROR: {error_msg} ---")
        return f"**Error:** {error_msg}"

def extract_text_from_document(file_path, mime_type):
    """Extracts text with detailed logging, including OCR fallback for PDFs."""
    if mime_type == 'application/pdf':
        try:
            print(f"--- [Text Extractor] Attempting direct text extraction from PDF: {os.path.basename(file_path)} ---")
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc).strip()
            if len(text) < 100:
                print("--- [Text Extractor] Low text content detected. Falling back to OCR... ---")
                return pytesseract.image_to_string(Image.open(file_path))
            return text
        except Exception as e:
            print(f"--- [Text Extractor] Direct extraction failed ({e}). Falling back to OCR... ---")
            return pytesseract.image_to_string(Image.open(file_path))
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return docx2txt.process(file_path)
    elif mime_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def process_input(file_path, text_query, vectorstore, llm_config, llm):
    """Acts as the main smart router for a single incoming file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('image/'):
        # Note: The UI layer now bundles multiple images into a single call to handle_vision_input
        return handle_vision_input([file_path], text_query, vectorstore, llm_config, llm)
    elif mime_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']:
        return handle_document_input(file_path, vectorstore)
    else:
        return f"Sorry, the file type '{mime_type}' is not supported yet."