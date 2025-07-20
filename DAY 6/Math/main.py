import os
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from word2number import w2n

# -------------------------------
# 1. Load API Key from .env
# -------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")

# -------------------------------
# 2. Define State
# -------------------------------
class AgentState(Dict[str, Any]):
    query: str
    result: str

# -------------------------------
# 3. Math Functions
# -------------------------------
def plus(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        return "Error: Division by zero!"
    return a / b

# -------------------------------
# 4. Initialize Gemini LLM
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=API_KEY
)

# -------------------------------
# 5. Detect Intent
# -------------------------------
def detect_intent(state: AgentState) -> str:
    query = state["query"].lower()
    if any(op in query for op in ["plus", "add", "+", "sum"]):
        return "add"
    elif any(op in query for op in ["minus", "subtract", "-"]):
        return "subtract"
    elif any(op in query for op in ["multiply", "times", "*"]):
        return "multiply"
    elif any(op in query for op in ["divide", "/"]):
        return "divide"
    else:
        return "general"

# -------------------------------
# 6. Math Handlers
# -------------------------------
def extract_numbers(text):
    # Remove punctuation and lowercase
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    numbers = []
    
    for word in clean_text.split():
        # If it's a digit or decimal
        if word.replace('.', '', 1).isdigit():
            numbers.append(float(word))
        else:
            # Try converting words to numbers (e.g., "five" -> 5)
            try:
                num = w2n.word_to_num(word)
                numbers.append(float(num))
            except:
                continue
    return numbers

def handle_add(state: AgentState):
    nums = extract_numbers(state["query"])
    if len(nums) == 2:
        state["result"] = f"Result: {plus(nums[0], nums[1])}"
    else:
        state["result"] = "Couldn't parse numbers properly."
    return state

def handle_subtract(state: AgentState):
    nums = extract_numbers(state["query"])
    if len(nums) == 2:
        state["result"] = f"Result: {subtract(nums[0], nums[1])}"
    else:
        state["result"] = "Couldn't parse numbers properly."
    return state

def handle_multiply(state: AgentState):
    nums = extract_numbers(state["query"])
    if len(nums) == 2:
        state["result"] = f"Result: {multiply(nums[0], nums[1])}"
    else:
        state["result"] = "Couldn't parse numbers properly."
    return state

def handle_divide(state: AgentState):
    nums = extract_numbers(state["query"])
    if len(nums) == 2:
        state["result"] = f"Result: {divide(nums[0], nums[1])}"
    else:
        state["result"] = "Couldn't parse numbers properly."
    return state

# -------------------------------
# 7. General Query Handler
# -------------------------------
def handle_general(state: AgentState):
    response = llm.invoke(state["query"])
    state["result"] = response.content
    return state

# -------------------------------
# 8. Build LangGraph
# -------------------------------
graph = StateGraph(AgentState)

graph.add_node("add", handle_add)
graph.add_node("subtract", handle_subtract)
graph.add_node("multiply", handle_multiply)
graph.add_node("divide", handle_divide)
graph.add_node("general", handle_general)

graph.set_entry_point("general")

# Router function
def router(state: AgentState) -> str:
    return detect_intent(state)

graph.add_conditional_edges("general", router, {
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "general": END
})

graph.add_edge("add", END)
graph.add_edge("subtract", END)
graph.add_edge("multiply", END)
graph.add_edge("divide", END)

app = graph.compile()

# -------------------------------
# 9. Run the Agent
# -------------------------------
if __name__ == "__main__":
    print("✅ AI Math & General Agent Ready! (type 'exit' to quit)")
    while True:
        query = input("\nAsk me anything: ")
        if query.lower() == "exit":
            break
        result = app.invoke({"query": query})
        print(result["result"])
