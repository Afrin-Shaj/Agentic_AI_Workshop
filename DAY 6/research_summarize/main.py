import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, Any
import json

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------------------
# 1. Load API Keys
# -------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Add it in .env file")

# -------------------------------
# 2. Define State
# -------------------------------
class AgentState(Dict[str, Any]):
    query: str
    info: str
    final_summary: str
    route: str

# -------------------------------
# 3. Initialize LLM
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0.3
)

# -------------------------------
# 4. RAG Setup with Sample Knowledge Base
# -------------------------------
sample_docs = [
    Document(page_content="LangGraph is a Python library for building stateful, multi-actor applications with LLMs. It's designed to create agent and multi-agent workflows."),
    Document(page_content="LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents."),
    Document(page_content="FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Retrieval-Augmented Generation (RAG) combines pre-trained language models with external knowledge retrieval."),
    Document(page_content="Vector databases store high-dimensional vectors and enable similarity search for semantic retrieval."),
    Document(page_content="Machine learning models can be fine-tuned using techniques like LoRA (Low-Rank Adaptation) and QLoRA."),
    Document(page_content="Transformer architecture revolutionized natural language processing with attention mechanisms."),
    Document(page_content="GPT models use autoregressive generation to produce human-like text responses."),
    Document(page_content="Embeddings convert text into numerical representations that capture semantic meaning."),
    Document(page_content="Prompt engineering involves crafting effective inputs to get desired outputs from language models.")
]

# Split documents
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(sample_docs)

# Create embeddings and vector store
embeddings = FakeEmbeddings(size=768)
vector_store = FAISS.from_documents(docs, embeddings)

# -------------------------------
# 5. Agent Functions
# -------------------------------

def router_agent(state: AgentState) -> AgentState:
    """Determines the routing based on query content"""
    query = state["query"].lower()
    
    # Check for time-sensitive queries
    time_keywords = ["latest", "news", "today", "current", "recent", "update", 
                    "now", "this week", "yesterday", "breaking"]
    
    # Check for RAG-related queries
    rag_keywords = ["langgraph", "faiss", "rag", "langchain", "vector", "embedding", 
                   "transformer", "gpt", "machine learning", "retrieval"]
    
    if any(word in query for word in time_keywords):
        state["route"] = "web_research"
    elif any(word in query for word in rag_keywords):
        state["route"] = "rag"
    else:
        state["route"] = "llm"
    
    print(f"🎯 Route determined: {state['route']}")
    return state

def llm_agent(state: AgentState) -> AgentState:
    """Handles general queries using LLM"""
    try:
        print(f"🤖 Processing with LLM: {state['query']}")
        response = llm.invoke(state["query"])
        state["info"] = response.content
    except Exception as e:
        state["info"] = f"Error processing with LLM: {str(e)}"
    return state

def web_research_agent(state: AgentState) -> AgentState:
    """Performs web research using Tavily or fallback web scraping"""
    print(f"🔍 Performing web research for: {state['query']}")
    
    try:
        if TAVILY_API_KEY:
            # Use Tavily if API key is available
            tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
            results = tool.run(state["query"])
            
            if results:
                if isinstance(results, list):
                    state["info"] = "\n\n".join([str(result) for result in results])
                else:
                    state["info"] = str(results)
            else:
                state["info"] = "No recent information found via Tavily search."
        else:
            # Fallback to basic web scraping
            state["info"] = fallback_web_search(state["query"])
            
    except Exception as e:
        print(f"⚠️ Web research error: {e}")
        state["info"] = f"Unable to perform web research at this time. Error: {str(e)}"
    
    return state

def fallback_web_search(query: str) -> str:
    """Fallback web search using requests and BeautifulSoup"""
    try:
        # Simple Google search simulation (for demo purposes)
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Note: This is a simplified example. In production, use proper APIs
        return f"Performed fallback search for '{query}'. For real-time results, please configure Tavily API key."
        
    except Exception as e:
        return f"Fallback search failed: {str(e)}"

def rag_agent(state: AgentState) -> AgentState:
    """Retrieves relevant information from knowledge base"""
    try:
        print(f"📚 Searching knowledge base for: {state['query']}")
        query = state["query"]
        
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=3)
        
        if docs:
            combined_info = "\n\n".join([f"Knowledge {i+1}: {d.page_content}" 
                                       for i, d in enumerate(docs)])
            state["info"] = combined_info
        else:
            state["info"] = "No relevant information found in knowledge base."
            
    except Exception as e:
        state["info"] = f"Error accessing knowledge base: {str(e)}"
    
    return state

def summarization_agent(state: AgentState) -> AgentState:
    """Creates structured summary of gathered information"""
    try:
        print("📝 Generating final summary...")
        
        route_info = f"Information source: {state.get('route', 'unknown')}"
        
        prompt = f"""
        Based on the following information, create a well-structured and comprehensive summary:

        Query: {state['query']}
        Source: {state.get('route', 'unknown')}
        
        Information:
        {state['info']}
        
        Please provide:
        1. A clear and concise answer to the query
        2. Key points and relevant details
        3. Any important context or implications
        
        Format the response in a professional and readable manner.
        """
        
        response = llm.invoke(prompt)
        state["final_summary"] = response.content
        
    except Exception as e:
        state["final_summary"] = f"Error generating summary: {str(e)}\n\nRaw info: {state.get('info', 'No information available')}"
    
    return state

# -------------------------------
# 6. Build LangGraph Workflow
# -------------------------------
def create_workflow():
    """Creates and compiles the LangGraph workflow"""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("router", router_agent)
    graph.add_node("llm", llm_agent)
    graph.add_node("web_research", web_research_agent)
    graph.add_node("rag", rag_agent)
    graph.add_node("summarize", summarization_agent)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional routing from router
    def route_decision(state: AgentState) -> str:
        return state["route"]
    
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "web_research": "web_research",
            "rag": "rag", 
            "llm": "llm"
        }
    )
    
    # All paths lead to summarization
    graph.add_edge("web_research", "summarize")
    graph.add_edge("rag", "summarize")
    graph.add_edge("llm", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()

# -------------------------------
# 7. Main Application
# -------------------------------
def main():
    """Main application loop"""
    print("✅ Research & Summarization Agent Ready!")
    print("📋 This agent can:")
    print("   • Answer general questions using LLM")
    print("   • Search the web for latest information")
    print("   • Retrieve from knowledge base (AI/ML topics)")
    print("   • Provide structured summaries")
    print("\n💡 Try queries like:")
    print("   • 'Latest AI news today' (web research)")
    print("   • 'How does RAG work?' (knowledge base)")
    print("   • 'Explain quantum computing' (LLM)")
    print("\nType 'exit' to quit\n")
    
    app = create_workflow()
    
    while True:
        try:
            query = input("🔍 Ask your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not query:
                print("Please enter a valid query.")
                continue
            
            print(f"\n🚀 Processing: '{query}'")
            print("-" * 50)
            
            # Initialize state
            initial_state = {
                "query": query,
                "info": "",
                "final_summary": "",
                "route": ""
            }
            
            # Run the workflow
            result = app.invoke(initial_state)
            
            print("\n📌 Final Summary:")
            print("=" * 50)
            print(result["final_summary"])
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again with a different query.")

# -------------------------------
# 8. Environment Setup Helper
# -------------------------------
def setup_environment():
    """Helper function to set up environment variables"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("📁 Creating .env file template...")
        with open(env_file, "w") as f:
            f.write("# API Keys for Research Agent\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
            f.write("TAVILY_API_KEY=your_tavily_api_key_here  # Optional for web search\n")
        
        print(f"✅ Created {env_file}")
        print("📝 Please add your API keys to the .env file")
        return False
    
    return True

# -------------------------------
# 9. Run Application
# -------------------------------
if __name__ == "__main__":
    try:
        if setup_environment() and GOOGLE_API_KEY:
            main()
        else:
            print("❌ Please configure your API keys in the .env file before running.")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("Please check your API keys and dependencies.")