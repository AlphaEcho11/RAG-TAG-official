# verify_env.py
import gradio
import sys

print("--- Environment Verification ---")
print(f"Python Executable: {sys.executable}")
print(f"Gradio Version: {gradio.__version__}")
print(f"Gradio Path: {gradio.__file__}")
print("----------------------------")