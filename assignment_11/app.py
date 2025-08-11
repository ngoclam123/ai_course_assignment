
import os
import json
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any
from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv("../.env")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Assistant with Weather & Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

@st.cache_resource
def initialize_agent():
    """Initialize the LangChain agent with tools"""
    try:
        # Initialize tools
        weather_api = OpenWeatherMapAPIWrapper()
        
        @tool
        def get_weather(city: str) -> str:
            """Get current weather for a city"""
            try:
                return f"Weather for {city}: {weather_api.run(city)}"
            except Exception as e:
                return f"Error getting weather for {city}: {str(e)}"
        
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="advanced",
            include_answer=True
        )
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-07-01-preview",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create agent
        tools = [get_weather, tavily_search]
        agent = create_react_agent(llm, tools)
        
        return agent, "✅ Agent initialized successfully"
    except Exception as e:
        return None, f"❌ Agent initialization failed: {str(e)}"

def main():
    st.title("🤖 AI Assistant with Weather & Search")
    st.markdown("*Ask me about weather conditions or search for current information!*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize agent
        if st.button("🔄 Initialize Agent"):
            with st.spinner("Initializing AI agent..."):
                agent, status = initialize_agent()
                st.session_state.agent = agent
                if agent:
                    st.success(status)
                else:
                    st.error(status)
        
        # Agent status
        if st.session_state.agent:
            st.success("🟢 Agent Ready")
        else:
            st.warning("🟡 Agent Not Initialized")
        
        # Chat controls
        st.markdown("---")
        st.header("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Export chat history
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="📥 Export Chat",
                data=chat_export,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Sample queries
        st.markdown("---")
        st.header("💡 Sample Queries")
        sample_queries = [
            "Weather in Paris",
            "Latest AI news",
            "Weather in Tokyo + cherry blossoms",
            "Sustainable energy trends 2025",
            "Climate in Australia"
        ]
        
        for query in sample_queries:
            if st.button(f"💬 {query}", key=f"sample_{query}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"🕒 {message['timestamp']}")
        
        # Chat input
        if prompt := st.chat_input("Ask me about weather or search for information..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            if st.session_state.agent:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            response = st.session_state.agent.invoke({
                                "messages": [("human", prompt)]
                            })
                            
                            if 'messages' in response and response['messages']:
                                ai_response = response['messages'][-1].content
                                st.markdown(ai_response)
                                
                                # Add AI response to session state
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": ai_response,
                                    "timestamp": datetime.now().isoformat()
                                })
                            else:
                                error_msg = "Sorry, I couldn't generate a response."
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": datetime.now().isoformat()
                            })
            else:
                st.error("❌ Please initialize the agent first using the sidebar")
    
    with col2:
        st.header("📊 Chat Statistics")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        st.metric("💬 Total Messages", total_messages)
        st.metric("👤 User Messages", user_messages)
        st.metric("🤖 AI Messages", ai_messages)
        
        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            st.markdown(f"**🕒 Last Activity:**\n{latest_message['timestamp']}")

if __name__ == "__main__":
    main()
