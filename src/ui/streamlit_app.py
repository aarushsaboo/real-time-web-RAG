import streamlit as st
import os
from src.config import API_KEY, MAX_SEARCH_RESULTS, MAX_CHUNKS_TO_RETRIEVE
from langchain_google_genai import GoogleGenerativeAI

def setup_page():
    st.set_page_config(page_title="Real-time Web RAG", page_icon="🌐", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.title("🌐 Real-time Web RAG with Gemini")
    st.markdown("This application performs real-time web search to answer your questions with up-to-date information.")

def setup_sidebar():
    with st.sidebar:
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