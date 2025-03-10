from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from src.config import API_KEY, LLM_MODEL

def generate_response(query, documents):
    """
    Generate a response to the user's query using the retrieved documents.
    
    Args:
        query (str): The user's query
        documents (list): The retrieved documents
        
    Returns:
        str: The generated response
    """
    # Initialize the model
    llm = GoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
    
    # Create context from documents
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Create prompt template
    prompt = PromptTemplate(
        template="""You are a helpful assistant that provides accurate information based on web search results.
        Answer the following query based on the provided context. If the context doesn't contain
        relevant information to answer the query, acknowledge that and suggest what the user might search for instead.
        
        Query: {query}
        
        Context:
        {context}
        
        Response:""",
        input_variables=["query", "context"]
    )
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Generate response
    response = chain.invoke({"query": query, "context": context})
    
    return response