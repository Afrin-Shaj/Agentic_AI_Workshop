# 🏪 Business Competitor Analysis AI

AI-powered assistant to analyze competitors, footfall trends, and estimate setup costs for small businesses in specific locations using real-time web data and generative AI.

---

## 🚀 Features

- 🔍 **Competitor Analysis:** Identifies key competitors based on location and business type.
- 💰 **Cost Estimation:** Estimates rent, setup costs, inventory, and operating expenses.
- 📊 **Footfall Analysis:** Analyzes peak hours and customer flow patterns.
- 🧠 **Intelligent Clarifications:** Asks follow-up questions if input lacks detail or confidence is low.
- 📂 **Session Tracking:** Maintains session-based memory and personalized response flow.
- 🧩 **Modular DAG-based Workflow:** Built using LangGraph for clear separation of logic and flow control.
- 💬 **Conversational Interface:** Uses Streamlit's chat UI for an interactive experience.

---

## 🛠️ How it was built

This project was designed using modern GenAI techniques combined with a multi-agent system and real-time web search capabilities. Here's how it all fits together:

### 🧱 1. LangGraph for Workflow Orchestration
- We built a **Directed Acyclic Graph (DAG)** using [LangGraph](https://github.com/langchain-ai/langgraph) to define the flow of actions:
  - Input → Search → Analyze → Respond
- Each step is a LangChain-powered node that modifies a shared `AgentState`.

### 🗣️ 2. LangChain + Gemini for Reasoning
- We used **Gemini 1.5 Flash** via `langchain-google-genai` for fast and smart reasoning.
- It handles:
  - Extracting structured info (location, business type)
  - Summarizing search results
  - Generating final analysis reports

### 🌐 3. Tavily for Real-Time Web Search
- We integrated the [Tavily Search API](https://www.tavily.com) to fetch up-to-date data from the web.
- Query examples:
  - `"rent for commercial shops in Koramangala"`
  - `"peak footfall hours for restaurants in Bandra"`

### 🧠 4. Custom LangChain Tools
We built custom tools like:
- `extract_location_info`: Extracts location + intent using the LLM
- `generate_search_queries`: Crafts intelligent search queries
- `analyze_competitor_data` and `analyze_cost_estimation`: LLMs create structured business reports

### 💬 5. Streamlit Chat UI
- A sleek and interactive interface lets users ask questions in natural language.
- Displays step-by-step conversation and formatted final reports.

---

## 📦 How to Run Locally

### Prerequisites:
- Python 3.9+
- API keys for:
  - Google Generative AI (`GOOGLE_API_KEY`)
  - Tavily (`TAVILY_API_KEY`)

### Steps:

```bash
git clone https://github.com/your-username/competitor-analysis-ai.git
cd competitor-analysis-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
