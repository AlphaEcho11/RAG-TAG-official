# main.py
import subprocess
import sys
from ui import app # Imports the demo object from ui/app.py

def main():
    print("--- Starting RAG-TAG Production System ---")

    # 1. Start the Huey Curator as a background process
    print("--- Launching background Curator process... ---")
    consumer_command = [sys.executable, '-m', 'huey.bin.huey_consumer', 'curator.tasks.huey']
    consumer_process = subprocess.Popen(consumer_command)
    print("--- Curator is online. ---")

    # 2. Launch the Gradio Web UI
    print("--- Launching Gradio Console... ---")
    # This will block and run the web server
    app.demo.launch()

    # 3. Clean up the background process when the UI is closed
    print("--- Shutting down Curator process... ---")
    consumer_process.terminate()
    consumer_process.wait()
    print("--- System offline. ---")

if __name__ == "__main__":
    main()