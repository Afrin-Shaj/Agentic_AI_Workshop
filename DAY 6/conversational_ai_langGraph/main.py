import os
import json
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from datetime import datetime
from dotenv import load_dotenv
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# Streamlit imports
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: UNDERSTANDING LANGGRAPH STATE
# ============================================================================
class AgentState(TypedDict):
    """
    Enhanced state structure for our conversational AI.
    This is the "memory" that flows through all nodes in our graph.
    """
    # Core conversation data
    messages: Annotated[List, add_messages]  # Full conversation history
    
    # User context
    user_location: Optional[str]  # Where the user's business is located
    business_type: str  # Type of business (clothing store, restaurant, etc.)
    analysis_type: str  # What kind of analysis they want
    
    # Processing data
    search_results: Dict[str, Any]  # Raw search results from tools
    processed_data: Dict[str, Any]  # Cleaned and structured data
    analysis_report: str  # Final formatted report
    
    # Flow control
    current_step: str  # Track progress through the workflow
    needs_clarification: bool  # Whether we need more info from user
    error_message: Optional[str]  # Any errors that occurred
    
    # Metadata
    session_id: str  # Unique session identifier
    timestamp: str  # When the conversation started

# ============================================================================
# STEP 2: ENVIRONMENT AND CONFIGURATION SETUP
# ============================================================================
class Config:
    """Configuration manager with validation"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.validate_config()
    
    def validate_config(self):
        """Validate that all required API keys are present"""
        missing_keys = []
        
        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")
        if not self.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        logger.info("✅ All API keys validated successfully")

# Initialize configuration
config = Config()

# ============================================================================
# STEP 3: LANGUAGE MODEL AND TOOLS SETUP
# ============================================================================
# Initialize the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,  # Balanced creativity and consistency
    max_tokens=2000,  # Longer responses for detailed analysis
    google_api_key=config.google_api_key
)

# Initialize search tool
search_tool = TavilySearchResults(
    max_results=8,  # More results for comprehensive analysis
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    tavily_api_key=config.tavily_api_key
)

# ============================================================================
# STEP 4: CUSTOM TOOLS FOR BUSINESS ANALYSIS
# ============================================================================
@tool
def extract_location_info(user_message: str) -> Dict[str, Any]:
    """
    Extract location and business information from user message.
    Uses the LLM to intelligently parse user input.
    """
    extraction_prompt = f"""
    Extract the following information from this user message: "{user_message}"
    
    Please identify:
    1. Location (city, area, neighborhood, or "unknown" if not specified)
    2. Business type (clothing store, restaurant, cafe, general shop, etc., or "unknown" if not specified)
    3. Analysis type - determine what the user actually wants:
       - "cost_estimation" if asking about rent, costs, investment, setup costs
       - "competitor_analysis" if asking about competitors, market analysis
       - "footfall_analysis" if asking about customer traffic, peak hours
       - "market_research" if asking about general market conditions
    4. Specific requirements or questions
    
    Return as JSON format:
    {{
        "location": "extracted location or 'unknown'",
        "business_type": "extracted business type or 'general_shop'",
        "analysis_type": "extracted analysis type or 'competitor_analysis'",
        "specific_requirements": ["list", "of", "requirements"],
        "confidence": 0.8
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        # Parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "location": "unknown",
                "business_type": "clothing store",
                "analysis_type": "competitor analysis",
                "specific_requirements": [],
                "confidence": 0.3
            }
    except Exception as e:
        logger.error(f"Error in location extraction: {e}")
        return {
            "location": "unknown",
            "business_type": "clothing store",
            "analysis_type": "competitor analysis",
            "specific_requirements": [],
            "confidence": 0.1
        }

@tool
def analyze_competitor_data(search_results: str, location: str, business_type: str) -> str:
    """
    Comprehensive competitor analysis using search results.
    This is where the real business intelligence happens.
    """
    analysis_prompt = f"""
    You are a senior business analyst specializing in retail market research.
    
    CONTEXT:
    - Location: {location}
    - Business Type: {business_type}
    - Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
    
    SEARCH DATA:
    {search_results}
    
    REQUIRED ANALYSIS:
    Please provide a comprehensive competitor analysis report with the following sections:
    
    1. EXECUTIVE SUMMARY
       - Key findings in 2-3 sentences
       - Overall market competitiveness rating (1-10)
    
    2. COMPETITOR LANDSCAPE
       - Major competitors identified
       - Market positioning of each
       - Unique selling propositions
    
    3. FOOTFALL & TIMING ANALYSIS
       - Peak hours and days
       - Customer traffic patterns
       - Seasonal variations
    
    4. COMPETITIVE ADVANTAGES & GAPS
       - What competitors do well
       - Market gaps and opportunities
       - Areas for differentiation
    
    5. STRATEGIC RECOMMENDATIONS
       - Immediate action items (next 30 days)
       - Medium-term strategies (3-6 months)
       - Long-term positioning (1 year+)
    
    6. RISK ASSESSMENT
       - Potential threats
       - Market saturation level
       - Entry barriers
    
    FORMAT: Professional business report with clear headings and bullet points.
    TONE: Analytical, data-driven, actionable
    LENGTH: Comprehensive but concise (aim for clarity over length)
    """
    
    try:
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in competitor analysis: {e}")
        return f"Error generating analysis: {str(e)}"

@tool
def analyze_cost_estimation(search_results: str, location: str, business_type: str) -> str:
    """
    Comprehensive cost estimation analysis for business setup.
    """
    cost_analysis_prompt = f"""
    You are a business consultant specializing in retail business setup and cost estimation.
    
    CONTEXT:
    - Location: {location}
    - Business Type: {business_type}
    - Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
    
    SEARCH DATA:
    {search_results}
    
    REQUIRED COST ANALYSIS:
    Please provide a comprehensive cost estimation report with the following sections:
    
    1. RENTAL COSTS
       - Average rent per sq ft in the area
       - Security deposit requirements
       - Monthly rental range for different shop sizes
    
    2. SETUP COSTS
       - Interior design and renovation
       - Equipment and fixtures
       - Licensing and permits
       - Initial marketing and branding
    
    3. INVENTORY INVESTMENT
       - Initial stock investment based on business type
       - Working capital requirements
       - Supplier terms and payment cycles
    
    4. OPERATIONAL COSTS (Monthly)
       - Staff salaries
       - Utilities (electricity, water, internet)
       - Maintenance and cleaning
       - Insurance and taxes
    
    5. TOTAL INVESTMENT BREAKDOWN
       - One-time setup costs
       - 6-month operational costs
       - Total capital requirement
    
    6. FINANCING OPTIONS
       - Bank loans available
       - Government schemes
       - ROI expectations and break-even analysis
    
    FORMAT: Provide specific numbers where possible, ranges when exact figures aren't available.
    TONE: Professional, practical, actionable
    Include disclaimers about market variations and the need for current market verification.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=cost_analysis_prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in cost analysis: {e}")
        return f"Error generating cost analysis: {str(e)}"

@tool
def generate_search_queries(location: str, business_type: str, analysis_type: str) -> List[str]:
    """
    Generate optimized search queries for comprehensive market research.
    Different types of analysis require different search strategies.
    """
    base_queries = []
    
    # Generate different queries based on analysis type
    if "cost_estimation" in analysis_type.lower():
        base_queries = [
            f"shop rent cost {location} commercial property",
            f"{business_type} investment cost {location}",
            f"retail shop setup cost {location} rent per sqft",
            f"{location} commercial property rent rates",
            f"{business_type} inventory stock cost India",
            f"business setup cost {location} {business_type}"
        ]
    elif "footfall" in analysis_type.lower():
        base_queries = [
            f"{location} {business_type} foot traffic patterns",
            f"busiest shopping hours {location}",
            f"{business_type} {location} peak customer times",
        ]
    elif "market" in analysis_type.lower():
        base_queries = [
            f"{location} retail market analysis {business_type}",
            f"{business_type} market trends {location}",
            f"{location} business opportunities {business_type}",
        ]
    else:  # competitor_analysis
        base_queries = [
            f"{business_type} competitors {location}",
            f"popular {business_type} {location} customer reviews",
            f"{business_type} {location} busy hours peak times",
        ]
    
    return base_queries[:6]  # Limit to 6 queries to avoid rate limits

# ============================================================================
# STEP 5: LANGGRAPH NODES - THE CORE PROCESSING LOGIC
# ============================================================================
async def input_processor_node(state: AgentState) -> AgentState:
    """
    NODE 1: Process and understand user input
    This node extracts intent, location, and business context
    """
    logger.info("🔍 Processing user input...")
    
    last_message = state["messages"][-1].content
    
    # Extract structured information using our custom tool
    extraction_result = extract_location_info.invoke({"user_message": last_message})
    
    # Update state with extracted information
    state["user_location"] = extraction_result.get("location", "unknown")
    state["business_type"] = extraction_result.get("business_type", "clothing store")
    state["analysis_type"] = extraction_result.get("analysis_type", "competitor analysis")
    
    # Determine if we need clarification
    confidence = extraction_result.get("confidence", 0.5)
    state["needs_clarification"] = confidence < 0.6 or state["user_location"] == "unknown"
    
    # Generate appropriate response
    if state["needs_clarification"]:
        if "cost" in last_message.lower() or "rent" in last_message.lower() or "investment" in last_message.lower():
            clarification_msg = """
I'd be happy to help you with business setup cost estimation! To provide accurate cost analysis, I need a bit more information:

🏢 **What type of shop are you planning to open?** (e.g., clothing store, grocery, electronics, cafe)
📍 **Which specific area in Koramangala?** (e.g., 5th Block, 7th Block, 100 Feet Road)
📏 **What size shop are you considering?** (approximate square feet)
💼 **What's your approximate budget range?** (helps me tailor recommendations)

The more specific you are, the better cost estimation I can provide!
            """
        else:
            clarification_msg = """
I'd be happy to help you with business analysis! To provide the most accurate insights, I need a bit more information:

🏢 **What type of business are you analyzing?** (e.g., clothing store, restaurant, cafe)
📍 **What's the specific location?** (e.g., Koramangala Bangalore, Connaught Place Delhi)
🎯 **What specific insights do you need?** (e.g., competitor analysis, cost estimation, market opportunities)

The more specific you are, the better analysis I can provide!
            """
        state["messages"].append(AIMessage(content=clarification_msg))
        state["current_step"] = "needs_clarification"
    else:
        if "cost_estimation" in state["analysis_type"]:
            confirmation_msg = f"""
Perfect! I'll help you estimate the costs for setting up a **{state['business_type']}** in **{state['user_location']}**.

💰 I'll analyze:
- Rental costs and property rates
- Initial setup and renovation expenses
- Inventory and stock investment
- Monthly operational costs
- Total capital requirements

Let me gather the current market data and prepare your cost estimation report...
            """
        else:
            confirmation_msg = f"""
Perfect! I'll analyze **{state['business_type']}** competitors in **{state['user_location']}**.

🔍 I'll search for:
- Key competitors in your area
- Customer footfall patterns
- Peak business hours
- Market opportunities

Let me gather the data and prepare your analysis report...
            """
        state["messages"].append(AIMessage(content=confirmation_msg))
        state["current_step"] = "ready_for_search"
    
    return state

async def search_coordinator_node(state: AgentState) -> AgentState:
    """
    NODE 2: Coordinate multiple searches for comprehensive data
    This node manages the search strategy and collects relevant data
    """
    logger.info("🔎 Coordinating search operations...")
    
    if state["needs_clarification"]:
        return state  # Skip search if we need more info
    
    # Generate optimized search queries
    search_queries = generate_search_queries.invoke({
        "location": state["user_location"],
        "business_type": state["business_type"],
        "analysis_type": state["analysis_type"]
    })
    
    all_results = []
    successful_searches = 0
    failed_searches = []
    
    # Execute searches with error handling
    for query in search_queries:
        try:
            logger.info(f"   Searching: {query}")
            results = search_tool.invoke({"query": query})
            if results:
                all_results.extend(results)
                successful_searches += 1
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            failed_searches.append({"query": query, "error": str(e)})
    
    # Store comprehensive search results
    state["search_results"] = {
        "queries_executed": search_queries,
        "results": all_results,
        "successful_searches": successful_searches,
        "failed_searches": failed_searches,
        "total_results": len(all_results),
        "search_timestamp": datetime.now().isoformat()
    }
    
    state["current_step"] = "search_completed"
    
    if successful_searches == 0:
        state["error_message"] = "Unable to retrieve search data. Will provide general analysis."
    
    return state

async def analysis_engine_node(state: AgentState) -> AgentState:
    """
    NODE 3: Generate comprehensive business analysis
    This is where the AI creates actionable business insights
    """
    logger.info("📊 Generating business analysis...")
    
    if state["needs_clarification"]:
        return state
    
    search_data = state["search_results"]
    analysis_type = state["analysis_type"]
    
    if not search_data.get("results") or search_data["successful_searches"] == 0:
        # Generate template analysis if no search data
        analysis = await generate_fallback_analysis(state)
    else:
        # Generate analysis using search results based on analysis type
        search_summary = json.dumps(search_data, indent=2)
        
        if "cost_estimation" in analysis_type.lower():
            analysis = analyze_cost_estimation.invoke({
                "search_results": search_summary,
                "location": state["user_location"],
                "business_type": state["business_type"]
            })
        else:
            # Default to competitor analysis
            analysis = analyze_competitor_data.invoke({
                "search_results": search_summary,
                "location": state["user_location"],
                "business_type": state["business_type"]
            })
    
    state["analysis_report"] = analysis
    state["current_step"] = "analysis_completed"
    
    return state

async def response_formatter_node(state: AgentState) -> AgentState:
    """
    NODE 4: Format the final response for optimal user experience
    This node creates the final, polished response
    """
    logger.info("📝 Formatting final response...")
    
    if state["needs_clarification"]:
        return state  # Response already added in input processor
    
    # Create comprehensive final response
    if state["analysis_report"]:
        final_response = format_analysis_response(state)
    else:
        final_response = "I encountered an issue generating your analysis. Please try again with more specific location details."
    
    state["messages"].append(AIMessage(content=final_response))
    state["current_step"] = "completed"
    
    return state

# ============================================================================
# STEP 6: HELPER FUNCTIONS
# ============================================================================

async def generate_fallback_analysis(state: AgentState) -> str:
    """Generate a template analysis when search data is unavailable"""
    location = state["user_location"]
    business_type = state["business_type"]
    analysis_type = state["analysis_type"]
    
    if "cost_estimation" in analysis_type.lower():
        fallback_prompt = f"""
        Create a comprehensive business setup cost estimation for {business_type} in {location}.
        
        Since real-time data isn't available, provide typical cost ranges based on your knowledge:
        1. Rental costs per sq ft in the area
        2. Initial setup and renovation costs
        3. Inventory investment requirements
        4. Monthly operational expenses
        5. Total capital requirement breakdown
        6. ROI expectations and break-even timeline
        
        Make it specific to {location} and {business_type} based on your knowledge of the area and business type.
        Include disclaimers about market variations.
        """
    else:
        # Default competitor analysis fallback
        fallback_prompt = f"""
        Create a comprehensive competitor analysis template for {business_type} in {location}.
        
        Since real-time data isn't available, provide:
        1. Typical competitor categories for this business type
        2. General market dynamics in similar locations
        3. Standard peak hours and footfall patterns
        4. Common competitive strategies
        5. Actionable recommendations
        
        Make it specific to {location} based on your knowledge of the area.
        """
    
    response = llm.invoke([HumanMessage(content=fallback_prompt)])
    return response.content

def format_analysis_response(state: AgentState) -> str:
    """Format the final analysis response with proper structure"""
    analysis = state["analysis_report"]
    location = state["user_location"]
    business_type = state["business_type"]
    analysis_type = state["analysis_type"]
    search_stats = state["search_results"]
    
    if "cost_estimation" in analysis_type.lower():
        # Format cost estimation response
        formatted_response = f"""
# 💰 Business Setup Cost Estimation

**Location:** {location}
**Business Type:** {business_type}
**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}
**Data Sources:** {search_stats.get('successful_searches', 0)} search queries

---

{analysis}

---

## 📋 Next Steps for Cost Planning

1. **Immediate Actions** (This Week)
   - Visit 3-5 potential locations and get exact rent quotes
   - Contact local property brokers for current market rates
   - Get quotes from interior designers and contractors

2. **Short-term Planning** (Next Month)
   - Finalize location and negotiate lease terms
   - Apply for business licenses and permits
   - Identify potential suppliers and negotiate payment terms

3. **Financial Preparation** (Next Quarter)
   - Arrange financing through banks or investors
   - Set up business accounts and accounting systems
   - Plan soft launch and marketing budget

**Important Disclaimer:** These are estimated costs based on available data. Actual costs may vary significantly based on current market conditions, specific location, and business requirements. Always verify with local sources and current market rates.

Would you like me to help you with competitor analysis for this location, or do you need more specific information about any cost component?
        """
    else:
        # Default competitor analysis format
        formatted_response = f"""
# 🏪 Competitor Analysis Report

**Location:** {location}
**Business Type:** {business_type}
**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}
**Data Sources:** {search_stats.get('successful_searches', 0)} search queries

---

{analysis}

---

## 📋 Next Steps

1. **Immediate Actions** (This Week)
   - Visit top 2-3 competitors during different times
   - Note their pricing, customer service, and store layout
   - Check their social media presence and engagement

2. **Short-term Research** (Next Month)
   - Conduct customer surveys about shopping preferences
   - Monitor competitor promotional activities
   - Analyze your own footfall patterns

3. **Strategic Planning** (Next Quarter)
   - Develop differentiation strategies based on identified gaps
   - Plan marketing campaigns for peak hours
   - Consider partnership opportunities

Would you like me to dive deeper into any specific aspect of this analysis, or would you like to explore cost estimation for setting up your business?
        """
    
    return formatted_response

# ============================================================================
# STEP 7: LANGGRAPH WORKFLOW DEFINITION
# ============================================================================
def create_workflow_graph():
    """
    Create the LangGraph workflow with all nodes and edges.
    This defines the conversation flow and decision logic.
    """
    logger.info("🏗️ Building LangGraph workflow...")
    
    # Initialize the graph with our state structure
    workflow = StateGraph(AgentState)
    
    # Add all nodes to the graph
    workflow.add_node("input_processor", input_processor_node)
    workflow.add_node("search_coordinator", search_coordinator_node)
    workflow.add_node("analysis_engine", analysis_engine_node)
    workflow.add_node("response_formatter", response_formatter_node)
    
    # Define the entry point
    workflow.set_entry_point("input_processor")
    
    # Add conditional routing logic
    def route_after_input(state: AgentState) -> str:
        """Decide whether to search or ask for clarification"""
        if state["needs_clarification"]:
            return "response_formatter"
        return "search_coordinator"
    
    def route_after_search(state: AgentState) -> str:
        """Always proceed to analysis after search"""
        return "analysis_engine"
    
    def route_after_analysis(state: AgentState) -> str:
        """Always format response after analysis"""
        return "response_formatter"
    
    # Add conditional edges (decision points)
    workflow.add_conditional_edges(
        "input_processor",
        route_after_input,
        {
            "search_coordinator": "search_coordinator",
            "response_formatter": "response_formatter"
        }
    )
    
    # Add simple edges (always go to next node)
    workflow.add_edge("search_coordinator", "analysis_engine")
    workflow.add_edge("analysis_engine", "response_formatter")
    workflow.add_edge("response_formatter", END)
    
    # Add memory for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph into an executable application
    app = workflow.compile(checkpointer=memory)
    
    logger.info("✅ LangGraph workflow compiled successfully")
    return app

# ============================================================================
# STEP 8: MAIN APPLICATION CLASS
# ============================================================================
class CompetitorAnalysisAI:
    """
    Enhanced Conversational AI for Competitor Analysis
    Built with LangGraph for robust conversation flow management
    """
    
    def __init__(self):
        self.app = create_workflow_graph()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🚀 CompetitorAnalysisAI initialized with session: {self.session_id}")
    
    async def chat(self, user_input: str) -> str:
        """
        Main chat interface - async for better performance
        """
        logger.info(f"👤 User input: {user_input[:100]}...")
        
        # Create configuration for this conversation thread
        config = RunnableConfig(
            configurable={"thread_id": self.session_id}
        )
        
        # Initialize state for new conversation or continue existing
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_location": "",
            "business_type": "clothing store",
            "analysis_type": "competitor analysis",
            "search_results": {},
            "processed_data": {},
            "analysis_report": "",
            "current_step": "start",
            "needs_clarification": False,
            "error_message": None,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Execute the graph
            result = await self.app.ainvoke(initial_state, config)
            
            # Return the last AI message
            if result["messages"]:
                return result["messages"][-1].content
            else:
                return "I'm having trouble processing your request. Please try again."
                
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        try:
            config = RunnableConfig(configurable={"thread_id": self.session_id})
            state = self.app.get_state(config)
            
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    }
                    for msg in messages
                ]
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        try:
            config = RunnableConfig(configurable={"thread_id": self.session_id})
            state = self.app.get_state(config)
            
            if state and state.values:
                return {
                    "session_id": self.session_id,
                    "current_step": state.values.get("current_step", "unknown"),
                    "location": state.values.get("user_location", "unknown"),
                    "business_type": state.values.get("business_type", "unknown"),
                    "message_count": len(state.values.get("messages", [])),
                    "last_activity": datetime.now().isoformat()
                }
            return {"session_id": self.session_id, "status": "new"}
        except Exception as e:
            logger.error(f"Error retrieving session info: {e}")
            return {"session_id": self.session_id, "status": "error"}

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Business Competitor Analysis AI",
        page_icon="🏪",
        layout="wide"
    )
    
    st.title("🏪 Business Competitor Analysis AI")
    st.markdown("""
    This AI helps business owners analyze competitors and estimate setup costs for their business.
    Ask questions like:
    - "What are the competitors for a coffee shop in Koramangala?"
    - "How much does it cost to open a clothing store in Connaught Place?"
    - "What are peak hours for restaurants in Bandra?"
    """)
    
    # Initialize session state
    if "ai" not in st.session_state:
        st.session_state.ai = CompetitorAnalysisAI()
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about competitors or business setup costs"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                response = asyncio.run(st.session_state.ai.chat(prompt))
            st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("Session Info")
        if st.session_state.ai:
            session_info = st.session_state.ai.get_session_info()
            st.json(session_info)
        
        st.header("Example Queries")
        st.markdown("""
        - "Analyze competitors for a bakery in Indiranagar"
        - "Estimate costs for opening a gym in Powai"
        - "What's the foot traffic like for electronics stores in Nehru Place?"
        """)
        
        if st.button("Clear Chat"):
            st.session_state.ai = CompetitorAnalysisAI()
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()