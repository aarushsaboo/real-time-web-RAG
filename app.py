import streamlit as st
import time
import os
import google.generativeai as genai

from src.search.web_search import search_web
from src.processing.scraper import process_urls
from src.retrieval.vector_store import create_vector_store, retrieve_documents
from src.generation.response_generator import generate_response
from src.config import API_KEY
from ui.streamlit_app import setup_page, setup_sidebar, initialize_api, display_chat_history, get_user_input

def real_time_web_rag(query, num_results, num_chunks):
    # Search the web
    with st.status("🔎 Searching the web...") as status:
        urls = search_web(query, num_results=num_results)
        if not urls:
            status.update(label="❌ Search failed", state="error")
            return "Unable to find relevant information from web search."
        status.update(label=f"✅ Found {len(urls)} relevant pages", state="complete")
    
    # Process web pages
    with st.status("📄 Processing web pages...") as status:
        progress_bar = st.progress(0)
        texts, successful_urls = process_urls(urls, progress_bar.progress)
        if not texts:
            status.update(label="❌ Content extraction failed", state="error")
            return "Failed to extract content from the search results."
        status.update(label=f"✅ Processed {len(texts)} pages successfully", state="complete")
    
    # Display sources
    with st.expander("View sources"):
        for url in successful_urls:
            st.write(url)
    
    # Analyze content
    with st.status("🧠 Analyzing content...") as status:
        vector_store = create_vector_store(texts)
        if not vector_store:
            status.update(label="❌ Analysis failed", state="error")
            return "Failed to analyze the content."
        
        retrieved_docs = retrieve_documents(vector_store, query, k=num_chunks)
        if not retrieved_docs:
            status.update(label="❌ No relevant content found", state="error")
            return "No relevant information found in the processed content."
        status.update(label="✅ Analysis complete", state="complete")
    
    # Generate response
    with st.status("💬 Generating response...") as status:
        response = generate_response(query, retrieved_docs)
        status.update(label="✅ Response ready", state="complete")
    
    return response

def main():
    # Setup UI
    setup_page()
    num_results, num_chunks = setup_sidebar()
    initialize_api()
    
    # Display previous messages
    display_chat_history()
    
    # Get user input
    user_query = get_user_input()
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process the query and generate response
        with st.chat_message("assistant"):
            response = real_time_web_rag(user_query, num_results, num_chunks)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.caption("Real-time Web RAG prototype - Data refreshed with each query")

if __name__ == "__main__":
    main()