import streamlit as st
import os
import time
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from src.config import API_KEY, MAX_SEARCH_RESULTS, MAX_CHUNKS_TO_RETRIEVE
from src.memory.conversation_memory import EnhancedConversationMemory

def setup_page():
    st.set_page_config(page_title="Real-time Web RAG", page_icon="🌐", layout="wide")
    
    # Initialize chat sessions if not already done
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"default": {"messages": [], "memory": EnhancedConversationMemory()}}
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = "default"
    
    # For backward compatibility
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.title("🌐 Real-time Web RAG with Gemini")
    st.markdown("This application performs real-time web search to answer your questions with up-to-date information.")

def setup_sidebar():
    with st.sidebar:
        st.header("Chats")
        
        # Button to create new chat
        if st.button("➕ New Chat"):
            create_new_chat()
            st.rerun()
        
        # List all available chats
        for chat_id in st.session_state.chat_sessions:
            col1, col2 = st.columns([5, 1])
            
            # Display chat button (highlighted if current)
            chat_name = chat_id if chat_id == "default" else f"Chat {chat_id.split('_')[1]}"
            is_current = chat_id == st.session_state.current_chat_id
            
            with col1:
                if st.button(
                    f"**{chat_name}**" if is_current else chat_name,
                    key=f"btn_{chat_id}",
                    use_container_width=True
                ):
                    switch_chat(chat_id)
                    st.rerun()
            
            # Delete button (except for default chat)
            with col2:
                if chat_id != "default" and st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
        
        st.header("Configuration")
        st.success("API Key loaded successfully!")
        
        num_results = st.slider("Number of search results", 
                               min_value=3, max_value=10, value=MAX_SEARCH_RESULTS)
        
        num_chunks = st.slider("Number of text chunks to retrieve", 
                              min_value=2, max_value=8, value=MAX_CHUNKS_TO_RETRIEVE)
        
        st.header("About")
        st.markdown("""
        This app uses:
        - DuckDuckGo for web search
        - Google's Gemini 1.5 Flash for text generation
        - FAISS for vector search
        - BeautifulSoup for web scraping
        """)
        
    return num_results, num_chunks

def initialize_api():
    os.environ["GOOGLE_API_KEY"] = API_KEY

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_user_input():
    return st.chat_input("Ask a question...")

def create_new_chat():
    """Create a new chat session with a unique ID based on timestamp"""
    new_chat_id = f"chat_{int(time.time())}"
    st.session_state.chat_sessions[new_chat_id] = {
        "messages": [],
        "memory": EnhancedConversationMemory()  # Properly initialize the class for each new chat
    }
    st.session_state.current_chat_id = new_chat_id
    
def switch_chat(chat_id):
    """Switch to a different chat session"""
    if chat_id in st.session_state.chat_sessions:
        st.session_state.current_chat_id = chat_id

def delete_chat(chat_id):
    """Delete a chat session"""
    if chat_id in st.session_state.chat_sessions and len(st.session_state.chat_sessions) > 1:
        del st.session_state.chat_sessions[chat_id]
        # Switch to the first available chat
        st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions))


# other functions
def get_current_chat_data():
    """Get the messages and memory for the current chat"""
    chat_id = st.session_state.current_chat_id
    chat_data = st.session_state.chat_sessions[chat_id]
    return chat_data["messages"], chat_data["memory"]

def display_chat_history():
    messages, _ = get_current_chat_data()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])