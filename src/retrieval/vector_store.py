from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def create_vector_store(texts):
    if not texts:
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    vector_store = FAISS.from_texts(all_chunks, embeddings)
    return vector_store

def retrieve_documents(vector_store, query, k=4):
    if not vector_store:
        return []
    
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return retrieved_docs