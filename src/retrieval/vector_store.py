from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def create_vector_store(texts):
    """
    Create a vector store from a list of texts.
    
    Args:
        texts (list): A list of text strings
        
    Returns:
        FAISS: A FAISS vector store
    """
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=API_KEY,
            model=EMBEDDING_MODEL
        )
        
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        chunks = []
        for text in texts:
            chunks.extend(text_splitter.split_text(text))
        
        # Create vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        return vector_store
    
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def retrieve_documents(vector_store, query, k=4):
    """
    Retrieve relevant documents from the vector store.
    
    Args:
        vector_store (FAISS): A FAISS vector store
        query (str): The query to search for
        k (int): The number of documents to retrieve
        
    Returns:
        list: A list of retrieved documents
    """
    try:
        # Search for similar documents
        documents = vector_store.similarity_search(query, k=k)
        
        return documents
    
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None