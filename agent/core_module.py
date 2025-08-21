# agent/core_module.py
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# --- UPGRADED CORE PROMPT ---
core_system_prompt = ChatPromptTemplate.from_template(
"""
You are an expert system rigorously applying the CORE protocol. Analyze the user's query and history, then output a structured JSON plan for the agent to execute.

**Conversation History:**
{history}

**User's Latest Query:** "{question}"

**Your Instructions:**

**Part A: Situational Awareness & Goal Setting:**
1.  Analyze User's Need: What is the user's true goal?
2.  Identify Constraints: Are there any operational limitations?
3.  Formulate Adaptive Goal: State the most helpful, achievable goal.

**Part B: Action Recommendation:**
1.  Recommend a Tool: Based on the user's need, choose the single best tool. The available tools are:
    - "text_rag": For all text-based questions and information retrieval.
    - "image_generator": For explicit requests to create or draw an image.
    - "request_file": Use this if the user's query requires a document, file, or image that has not been provided.

2.  Define Tool Parameters: Specify the necessary parameters for the chosen tool.
    - For "text_rag", the parameter is the "query".
    - For "image_generator", it is the "prompt".
    - For "request_file", the parameter is a "request_message" (a friendly string asking the user for the file).

**Output Format:**
Provide your analysis as a single, valid JSON object.

{{
    "situational_awareness": {{
        "user_need": "...",
        "operational_constraints": "...",
        "adaptive_goal": "..."
    }},
    "recommended_action": {{
        "tool": "...",
        "parameters": {{...}}
    }}
}}
"""
)

# --- FUNCTION DEFINITION ---
def analyze_context(question: str, history: list, llm) -> dict:
    """
    Analyzes the user's query using the enhanced CORE protocol.
    """
    print("\n--- Activating CORE Module (Enhanced Protocol) ---")
    
    formatted_history = "\n".join([f"User: {h.get('query', '')}\nAgent: {h.get('response', '')}" for h in history])
    
    chain = core_system_prompt | llm | StrOutputParser()
    
    response_text = chain.invoke({
        "question": question,
        "history": formatted_history if formatted_history else "No history yet."
    })
    
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        core_analysis = json.loads(clean_json_text)
        print("--- CORE Analysis Complete ---")
        return core_analysis
    except json.JSONDecodeError:
        print("--- CORE Analysis Failed: Could not decode LLM response into JSON ---")
        return None