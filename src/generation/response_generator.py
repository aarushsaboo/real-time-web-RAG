from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from src.config import LLM_MODEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(query, retrieved_docs, conversation_history=None):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    
    # Format conversation history
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        # Get the last few conversations (limit to prevent context overflow)
        recent_history = conversation_history[-6:]  # Last 3 exchanges (3 user + 3 assistant messages)
        for message in recent_history[:-1]:  # Exclude the current query
            role = message["role"]
            content = message["content"]
            conversation_context += f"{role.capitalize()}: {content}\n\n"
    
    template = """
    You are a helpful assistant that answers questions based on the latest web search results while maintaining context from previous conversation.
    
    {conversation_history}
    
    Use ONLY the following web search results to answer the user's current question. If you don't know the answer based on these results, say so.
    Do not make up information or use your training data to answer.
    
    IMPORTANT: Do NOT include any document IDs, citations, or references like [Document(id='xxx')] in your response.
    Present the information in a clean, readable format without mentioning sources or document identifiers.
    
    When appropriate, remember and refer to information from our previous conversation.

    Web search results:
    {context}

    Answer any questions or queries properly. User's current question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {
            "context": lambda x: format_docs(x), 
            "question": RunnablePassthrough(),
            "conversation_history": lambda _: conversation_context
        }
        | prompt
        | llm
    )
    
    response = chain.invoke(retrieved_docs, {"question": query})
    return response.content