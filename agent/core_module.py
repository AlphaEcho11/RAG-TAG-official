# core_module.py
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize the same LLM you use in your agent
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gemma-3-1b-it-qat",
    temperature=0.3 
)

# --- ENHANCED PROMPT ---
# This prompt is a more rigorous, direct implementation of the official CORE System Prompt.
core_system_prompt = ChatPromptTemplate.from_template(
"""
You are an expert system rigorously applying the CORE protocol for every user query. Analyze the user's query and conversation history, then output a structured JSON plan.

**Conversation History:**
{history}

**User's Latest Query:** "{question}"

**Your Instructions:**

**A. Situational Awareness & Adaptive Goal Setting:**
1.  **Acknowledge Limitations:** Based on the query, note any operational constraints. Are you being asked for real-time information you can't access or to perform an action you can't do? [cite: 46]
2.  **Determine User's Underlying Need:** What is the user's true goal or intent? [cite: 47]
3.  **Formulate Adaptive Goal:** State the most helpful, achievable goal you can accomplish right now to meet that need within your genuine scope. [cite: 30, 48]

**B. Focused Execution & Calibrated Response:**
1.  **Define Achievable Objective(s):** List the concrete steps required to achieve the Adaptive Goal from Part A. [cite: 51]
2.  **Establish Relevance Hierarchy:** What keywords or concepts are most critical for the search and response? List them in order of importance. [cite: 53]
3.  **Advise on Transparency:** If you noted any significant limitations in A.1 that will affect the final response, set "articulate_limitations" to true. Otherwise, set it to false. [cite: 57, 59]

**Output Format:**
Provide your analysis as a single, valid JSON object.

{{
    "situational_awareness": {{
        "user_need": "...",
        "operational_constraints": "...",
        "adaptive_goal": "..."
    }},
    "focused_execution": {{
        "achievable_objectives": ["...", "..."],
        "relevance_hierarchy": ["...", "..."],
        "articulate_limitations": boolean
    }}
}}
"""
)

def analyze_context(question: str, history: list) -> dict:
    """
    Analyzes the user's query using the enhanced CORE protocol.
    """
    print("\n--- Activating CORE Module (Enhanced Protocol) ---")
    
    formatted_history = "\n".join([f"User: {h['query']}\nAgent: {h['response']}" for h in history])
    
    chain = core_system_prompt | llm
    
    response_text = chain.invoke({
        "question": question,
        "history": formatted_history if formatted_history else "No history yet."
    })
    
    try:
        clean_json_text = response_text.content.strip().replace("```json", "").replace("```", "")
        core_analysis = json.loads(clean_json_text)
        print("--- CORE Analysis Complete ---")
        return core_analysis
    except json.JSONDecodeError:
        print("--- CORE Analysis Failed: Could not decode LLM response into JSON ---")
        return None