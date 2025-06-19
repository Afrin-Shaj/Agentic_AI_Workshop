import streamlit as st
import os
import requests
import json
from typing import Optional, Dict, Any
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import time
import re

os.environ["GOOGLE_API_KEY"] = "AIzaSyB-3E8uLNEjRFyRV83bONU4K1rY3eqtMw0"
WEATHER_API_KEY = "f32fb789ed1c418ba7940207251906"  

class TravelAssistantConfig:
    """Configuration class to keep our API keys and settings organized"""
    def __init__(self):
        self.weather_api_key = WEATHER_API_KEY
        self.weather_base_url = "http://api.weatherapi.com/v1"
        self.weather_emojis = {
            "sunny": "☀️", "cloudy": "☁️", "rainy": "🌧️",
            "snowy": "❄️", "stormy": "⛈️", "clear": "🌤️"
        }

config = TravelAssistantConfig()

@tool
def get_weather_forecast(location: str) -> str:
    """
    Fetches current weather and 3-day forecast for a given location.
    
    Args:
        location (str): The city or location name (e.g., "Paris", "Ooty")
    
    Returns:
        str: Formatted weather information with current conditions and forecast
    """
    try:
        current_url = f"{config.weather_base_url}/current.json"
        forecast_url = f"{config.weather_base_url}/forecast.json"
        
        params = {
            "key": config.weather_api_key,
            "q": location,
            "aqi": "no"
        }
        
        current_response = requests.get(current_url, params=params)
        if current_response.status_code != 200:
            return f"Sorry, I couldn't fetch weather data for {location}. Please check the location name."
        
        current_data = current_response.json()
        
        forecast_params = params.copy()
        forecast_params["days"] = 3
        forecast_response = requests.get(forecast_url, params=forecast_params)
        if forecast_response.status_code != 200:
            return f"Got current weather but couldn't fetch forecast for {location}."
        
        forecast_data = forecast_response.json()
        
        location_name = current_data["location"]["name"]
        country = current_data["location"]["country"]
        current = current_data["current"]
        temp_c = current["temp_c"]
        temp_f = current["temp_f"]
        condition = current["condition"]["text"].lower()
        humidity = current["humidity"]
        wind_kph = current["wind_kph"]
        
        emoji = "🌡️"
        for key, value in config.weather_emojis.items():
            if key in condition:
                emoji = value
                break
        
        weather_report = f"""🏙️ Weather Report for {location_name}, {country}

{emoji} Current Conditions:
• Temperature: {temp_c}°C ({temp_f}°F)
• Conditions: {current['condition']['text']}
• Humidity: {humidity}%
• Wind Speed: {wind_kph} km/h

📅 3-Day Forecast:
"""
        
        for day in forecast_data["forecast"]["forecastday"]:
            date = day["date"]
            day_data = day["day"]
            max_temp = day_data["maxtemp_c"]
            min_temp = day_data["mintemp_c"]
            condition = day_data["condition"]["text"]
            weather_report += f"• {date}: {min_temp}°C - {max_temp}°C, {condition}\n"
        
        weather_report += f"\n💡 Travel Tip: Pack accordingly for the weather conditions!"
        return weather_report
        
    except requests.exceptions.RequestException as e:
        return f"Network error while fetching weather data: {str(e)}"
    except KeyError as e:
        return f"Unexpected response format from weather API: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def search_tourist_attractions(location: str) -> str:
    """
    Searches for top tourist attractions in a given location.
    
    Args:
        location (str): The city or location name (e.g., "Paris", "Ooty")
    
    Returns:
        str: Formatted list of tourist attractions with descriptions
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        search_queries = [
            f"top 10 tourist attractions {location} must visit places",
            f"best things to do in {location} travel guide"
        ]
        
        all_results = ""
        for i, query in enumerate(search_queries, 1):
            try:
                results = search_tool.run(query)
                all_results += f"\n{results}\n"
            except Exception as e:
                all_results += f"\nSearch failed: {str(e)}\n"
        
        if not all_results.strip():
            return f"I couldn't find specific tourist information for {location}. Try checking official tourism websites."
        
        formatted_response = f"""🗺️ Top Tourist Attractions in {location}

{all_results}

🎯 Pro Travel Tips:
• Check opening hours and ticket prices in advance
• Consider purchasing city tourist passes for discounts
• Book popular attractions online to skip lines
• Ask locals for hidden gems not in typical tourist guides!

Happy travels! 🧳✈️"""
        return formatted_response
        
    except Exception as e:
        return f"Sorry, I encountered an error while searching for attractions in {location}: {str(e)}"

@st.cache_resource
def initialize_travel_assistant():
    """Initialize the Travel Assistant AI with caching"""
    try:
        if not os.environ.get("GOOGLE_API_KEY") or os.environ["GOOGLE_API_KEY"] == "YOUR_GOOGLE_API_KEY":
            raise ValueError("Invalid Google API key. Please provide a valid key.")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            convert_system_message_to_human=True
        )
        
        tools = [get_weather_forecast, search_tourist_attractions]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent and friendly Travel Assistant AI. Your job is to help users plan their trips by providing:

1. Accurate weather information for their destination using the get_weather_forecast tool
2. Tourist attractions and places to visit using the search_tourist_attractions tool

IMPORTANT INSTRUCTIONS:
- ALWAYS use both tools for every destination query
- Present weather information first, then tourist attractions
- If a tool fails, mention it briefly but don't let it stop you from using the other tool
- Be enthusiastic and provide practical travel advice
- Format your response clearly with sections for weather and attractions
- Suggest what to pack based on weather conditions
- Keep responses organized and easy to read

You're helping someone plan an amazing trip!"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"Failed to initialize Travel Assistant: {str(e)}")
        return None

def parse_agent_response(response_text: str, destination: str):
    """
    Parse the agent response and extract weather and attractions sections
    """
    weather_content = []
    attractions_content = []
    general_content = []
    
    # Split by common patterns
    sections = re.split(r'(🌤️|🗺️|Weather|Attractions|Tourist)', response_text, flags=re.IGNORECASE)
    
    current_section = "general"
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Determine section type
        if any(keyword in section.lower() for keyword in ['weather', '🌤️', 'temperature', 'forecast', '°c', '°f']):
            current_section = "weather"
            weather_content.append(section)
        elif any(keyword in section.lower() for keyword in ['attractions', '🗺️', 'tourist', 'visit', 'temple', 'museum']):
            current_section = "attractions"
            attractions_content.append(section)
        else:
            if current_section == "weather":
                weather_content.append(section)
            elif current_section == "attractions":
                attractions_content.append(section)
            else:
                general_content.append(section)
    
    return weather_content, attractions_content, general_content

def display_travel_info(response_text: str, destination: str):
    """
    Display travel information with proper formatting
    """
    st.markdown("## 📋 Your Travel Information")
    st.markdown("---")
    
    # Try to get direct tool results as fallback
    try:
        weather_result = get_weather_forecast(destination)
        attractions_result = search_tourist_attractions(destination)
        
        # Display weather section
        st.markdown("### 🌤️ Weather Forecast")
        if "Sorry" not in weather_result and "error" not in weather_result.lower():
            st.info(weather_result)
        else:
            st.warning(f"⚠️ Weather information unavailable for {destination}. Please check a reliable weather app before traveling.")
        
        # Display attractions section
        st.markdown("### 🗺️ Tourist Attractions")
        if "Sorry" not in attractions_result and "error" not in attractions_result.lower():
            st.success(attractions_result)
        else:
            st.warning(f"⚠️ Could not retrieve specific attractions for {destination}. Try checking official tourism websites.")
        
        # Display agent's additional insights
        if response_text and len(response_text.strip()) > 100:
            st.markdown("### 💡 Additional Travel Insights")
            st.markdown(response_text)
            
    except Exception as e:
        st.error(f"Error displaying travel information: {str(e)}")
        # Fallback to showing raw response
        st.markdown("### 📝 Travel Information")
        st.info(response_text)

def main():
    st.set_page_config(
        page_title="🌍 Intelligent Travel Assistant",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .feature-box {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #1a1a1a;
    }
    .feature-box h4 {
        color: #333333;
        margin-bottom: 0.5rem;
    }
    .travel-tip {
        background: #f0fff0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .travel-tip h4 {
        color: #333333;
        margin-bottom: 0.5rem;
    }
    .stSuccess > div {
        background-color: #d4edda;
        color: #155724;
    }
    .stInfo > div {
        background-color: #cce7ff;
        color: #004085;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Intelligent Travel Assistant ✈️</h1>
        <p>Your AI-powered companion for weather forecasts and tourist attractions</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🛠️ Features")
        st.markdown("""
        <div class="feature-box">
            <h4>🌤️ Weather Forecasts</h4>
            <p>Real-time weather and 3-day forecasts for any destination</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h4>🗺️ Tourist Attractions</h4>
            <p>Discover top attractions and must-visit places</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h4>💡 Smart Recommendations</h4>
            <p>AI-powered travel tips and packing suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Examples")
        example_destinations = ["Paris", "Tokyo", "New York", "Coimbatore", "London", "Bali"]
        for dest in example_destinations:
            if st.button(f"🌍 {dest}", key=f"example_{dest}"):
                st.session_state.destination_input = dest
                st.rerun()
    
    if 'destination_input' not in st.session_state:
        st.session_state.destination_input = ""
    if 'travel_history' not in st.session_state:
        st.session_state.travel_history = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🗣️ Where would you like to travel?")
        
        with st.form("travel_query_form", clear_on_submit=False):
            destination = st.text_input(
                "Enter your destination:",
                value=st.session_state.destination_input,
                placeholder="e.g., Paris, Coimbatore, New York...",
                help="Enter any city or country name"
            )
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                submit_button = st.form_submit_button("🚀 Get Travel Info", type="primary")
            with col_b:
                clear_button = st.form_submit_button("🗑️ Clear")
        
        if clear_button:
            st.session_state.destination_input = ""
            st.rerun()
        
        if submit_button and destination:
            with st.spinner(f"🔍 Gathering travel information for {destination}..."):
                assistant = initialize_travel_assistant()
                
                if assistant:
                    try:
                        result = assistant.invoke({
                            "input": f"I want to visit {destination}. Please provide weather information and top tourist attractions."
                        })
                        
                        st.success("✅ Travel information retrieved successfully!")
                        
                        # Get the response text
                        response_text = result.get("output", "") if isinstance(result, dict) else str(result)
                        
                        # Display the information using our improved function
                        display_travel_info(response_text, destination)
                        
                        # Add to history
                        combined_response = f"Weather and attractions information for {destination}:\n\n{response_text}"
                        st.session_state.travel_history.append({
                            'destination': destination,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'response': combined_response
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error with AI agent: {str(e)}")
                        st.info(f"🔄 Trying direct tool access for {destination}...")
                        
                        # Direct tool fallback
                        try:
                            display_travel_info("", destination)
                            
                            st.session_state.travel_history.append({
                                'destination': destination,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'response': f"Direct tool results for {destination} (AI agent unavailable)"
                            })
                            
                        except Exception as fallback_e:
                            st.error(f"❌ All methods failed: {str(fallback_e)}")
                            st.info(f"Please try a different destination or check your internet connection. For {destination}, consider checking local weather apps and tourism websites directly.")
                else:
                    st.error("❌ Failed to initialize Travel Assistant. Please check your Google API key and refresh the page.")
        
        elif submit_button and not destination:
            st.warning("⚠️ Please enter a destination!")
    
    with col2:
        st.header("📚 Travel History")
        if st.session_state.travel_history:
            st.write(f"You've searched for {len(st.session_state.travel_history)} destinations:")
            for i, entry in enumerate(reversed(st.session_state.travel_history[-5:]), 1):
                with st.expander(f"🌍 {entry['destination']} - {entry['timestamp']}"):
                    st.text_area("Response:", entry['response'], height=200, key=f"history_{i}")
            if st.button("🗑️ Clear History"):
                st.session_state.travel_history = []
                st.rerun()
        else:
            st.info("No searches yet. Start by entering a destination above!")
        
        st.markdown("""
        <div class="travel-tip">
            <h4>💡 Travel Tips</h4>
            <ul>
                <li>Check visa requirements before traveling</li>
                <li>Book accommodations in advance</li>
                <li>Research local customs and etiquette</li>
                <li>Keep digital and physical copies of important documents</li>
                <li>Check travel advisories for your destination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🌟 Intelligent Travel Assistant - Powered by AI | Built with Streamlit & LangChain 🌟
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()