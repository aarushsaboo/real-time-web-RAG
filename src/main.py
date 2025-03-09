import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from duckduckgo_search import DDGS
import time
import concurrent.futures
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="Real-time Web RAG",
    page_icon="🌐",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🌐 Real-time Web RAG with Gemini")
st.markdown("""
This application performs real-time web search and retrieval to answer your questions 
with up-to-date information from the internet.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration")
    # Using the provided API key by default
    api_key = "AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA"
    st.success("API Key loaded successfully!")
    
    num_results = st.slider("Number of search results", min_value=3, max_value=10, value=5)
    num_chunks = st.slider("Number of text chunks to retrieve", min_value=2, max_value=8, value=4)
    
    st.header("About")
    st.markdown("""
    This app uses:
    - DuckDuckGo for web search
    - Google's Gemini 1.5 Flash for text generation
    - FAISS for vector search
    - BeautifulSoup for web scraping
    """)

# Initialize components
@st.cache_resource(show_spinner=False)
def initialize_components(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    return embeddings, llm, text_splitter

def search_web(query, num_results=5):
    """Search the web using DuckDuckGo and return URLs."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append(r['href'])
        return results
    except Exception as e:
        st.error(f"Error during web search: {e}")
        return []

def scrape_url(url):
    """Scrape content from a URL and return the extracted text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text and clean it
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        
        # Add source information
        text = f"Source: {url}\n\n{text}"
        
        return text
    except Exception as e:
        return ""

def process_urls(urls, progress_bar):
    """Process multiple URLs in parallel and return their content."""
    texts = []
    successful_urls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    texts.append(text)
                    successful_urls.append(url)
            except Exception as e:
                pass
            
            # Update progress
            progress_bar.progress((i + 1) / len(urls))
    
    return texts, successful_urls

def create_vector_store(texts, embeddings, text_splitter):
    """Create a vector store from the texts."""
    if not texts:
        return None
    
    # Split texts into chunks
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    # Create vector store
    vector_store = FAISS.from_texts(all_chunks, embeddings)
    return vector_store

def retrieve_from_vectorstore(vector_store, query, k=4):
    """Retrieve relevant documents from the vector store."""
    if not vector_store:
        return []
    
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return retrieved_docs

def format_docs(docs):
    """Format the retrieved documents for input to the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(query, retrieved_content, llm):
    """Generate a response using the LLM based on the retrieved content."""
    template = """
    You are a helpful assistant that answers questions based on the latest web search results.
    Use ONLY the following web search results to answer the user's question. If you don't know the answer based on these results, say so.
    Do not make up information or use your training data to answer.
    Include citations to the sources used in your answer.

    Web search results:
    {context}

    User question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": lambda x: format_docs(x), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    response = chain.invoke(retrieved_content, {"question": query})
    return response.content

def real_time_web_rag(query, api_key, num_results, num_chunks):
    # Initialize components
    embeddings, llm, text_splitter = initialize_components(api_key)
    
    # Step 1: Search the web
    with st.status("🔎 Searching the web...") as status:
        urls = search_web(query, num_results=num_results)
        if not urls:
            status.update(label="❌ Search failed", state="error")
            return "Unable to find relevant information from web search."
        status.update(label=f"✅ Found {len(urls)} relevant pages", state="complete")
    
    # Step 2: Scrape and process content
    with st.status("📄 Processing web pages...") as status:
        progress_bar = st.progress(0)
        texts, successful_urls = process_urls(urls, progress_bar)
        if not texts:
            status.update(label="❌ Content extraction failed", state="error")
            return "Failed to extract content from the search results."
        status.update(label=f"✅ Processed {len(texts)} pages successfully", state="complete")
    
    # Display sources
    with st.expander("View sources"):
        for url in successful_urls:
            st.write(url)
    
    # Step 3: Create vector store and retrieve content
    with st.status("🧠 Analyzing content...") as status:
        vector_store = create_vector_store(texts, embeddings, text_splitter)
        if not vector_store:
            status.update(label="❌ Analysis failed", state="error")
            return "Failed to analyze the content."
        
        retrieved_docs = retrieve_from_vectorstore(vector_store, query, k=num_chunks)
        if not retrieved_docs:
            status.update(label="❌ No relevant content found", state="error")
            return "No relevant information found in the processed content."
        status.update(label="✅ Analysis complete", state="complete")
    
    # Step 4: Generate response
    with st.status("💬 Generating response...") as status:
        response = generate_response(query, retrieved_docs, llm)
        status.update(label="✅ Response ready", state="complete")
    
    return response

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask a question...")

# Process user query
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Process the query and generate response
    with st.chat_message("assistant"):
        response = real_time_web_rag(user_query, api_key, num_results, num_chunks)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("Real-time Web RAG prototype - Data refreshed with each query")