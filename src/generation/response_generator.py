from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from src.config import LLM_MODEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(query, retrieved_docs):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    
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
    
    response = chain.invoke(retrieved_docs, {"question": query})
    return response.content