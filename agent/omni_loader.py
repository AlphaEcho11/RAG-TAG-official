# agent/omni_loader.py
import mimetypes
import os
import base64
import requests
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx2txt

# Import the memory-writing function from the main agent logic
from .agent_logic import write_novel_thought_to_vector_memory

def encode_image(image_path):
    """Encodes an image file into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def handle_vision_input(image_path, text_query, vectorstore, llm_config):
    """
    Sends an image to a multimodal LLM, gets an analysis, and then writes a
    text-based memory of the interaction to the vector store.
    """
    print(f"--- [Omni-Loader] Handling vision input for: {os.path.basename(image_path)} ---")
    
    try:
        base64_image = encode_image(image_path)
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": llm_config.get("model_name", "llava"), # Default to a common local model name
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1024
        }
        
        response = requests.post(f"{llm_config['base_url']}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        vision_response = response.json()['choices'][0]['message']['content']
        
        # Create and write the memory packet
        memory_content = f"The user uploaded an image ({os.path.basename(image_path)}) with the query: '{text_query}'. My analysis was: {vision_response}"
        print("--- [Omni-Loader] Writing vision interaction to vector memory... ---")
        write_novel_thought_to_vector_memory(vectorstore, f"Memory of analyzing image {os.path.basename(image_path)}", memory_content, "vision_log")
        
        return vision_response
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network Error: Could not connect to the vision model at {llm_config.get('base_url', 'the specified endpoint')}. Please ensure the service is running."
        print(f"--- [Omni-Loader] ERROR: {error_msg} ---")
        return f"**Error:** {error_msg}"
    except Exception as e:
        error_msg = f"An unexpected error occurred while analyzing the image: {e}"
        print(f"--- [Omni-Loader] ERROR: {error_msg} ---")
        return f"**Error:** {error_msg}"

def handle_document_input(file_path, vectorstore):
    """
    Extracts text from a document, writes it to vector memory, and returns a confirmation.
    """
    print(f"--- [Omni-Loader] Handling document input for: {os.path.basename(file_path)} ---")
    
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        extracted_text = extract_text_from_document(file_path, mime_type)

        if not extracted_text or not extracted_text.strip():
            return f"**Notice:** Could not extract any readable text from **{os.path.basename(file_path)}**."
            
        print(f"--- [Omni-Loader] Extracted {len(extracted_text)} characters. Writing to memory... ---")
        write_novel_thought_to_vector_memory(
            vectorstore,
            query=f"Content of document: {os.path.basename(file_path)}",
            response=extracted_text,
            scope="document_ingest"
        )
        return f"I have successfully read and memorized the document: **{os.path.basename(file_path)}**."
        
    except Exception as e:
        error_msg = f"An unexpected error occurred while processing the document. It might be corrupted or in an unsupported format. (Details: {e})"
        print(f"--- [Omni-Loader] ERROR: {error_msg} ---")
        return f"**Error:** {error_msg}"

def extract_text_from_document(file_path, mime_type):
    """Extracts text from various document types, with a fallback to OCR for PDFs."""
    if mime_type == 'application/pdf':
        try:
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc).strip()
            if len(text) < 100:
                print("--- [Text Extractor] Low text in PDF, trying OCR fallback. ---")
                return pytesseract.image_to_string(Image.open(file_path))
            return text
        except Exception as e:
            print(f"--- [Text Extractor] PDF extraction failed ({e}), falling back to OCR. ---")
            return pytesseract.image_to_string(Image.open(file_path))
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return docx2txt.process(file_path)
    elif mime_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def process_input(file_path, text_query, vectorstore, llm_config):
    """
    Acts as the main smart router for all incoming files.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type and mime_type.startswith('image/'):
        return handle_vision_input(file_path, text_query, vectorstore, llm_config)
    elif mime_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']:
        return handle_document_input(file_path, vectorstore)
    else:
        unsupported_msg = f"Sorry, the file type '{mime_type}' is not supported yet."
        print(f"--- [Omni-Loader] {unsupported_msg} ---")
        return unsupported_msg