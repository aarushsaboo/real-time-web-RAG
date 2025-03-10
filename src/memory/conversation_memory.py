from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL, API_KEY

class EnhancedConversationMemory:
    def __init__(self, max_token_limit=2000):
        # Use a summary buffer memory with a summarization model
        self.memory_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=API_KEY)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.memory_llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        
    def add_user_message(self, message):
        self.memory.chat_memory.add_user_message(message)
        
    def add_ai_message(self, message):
        self.memory.chat_memory.add_ai_message(message)
    
    def get_context(self):
        return self.memory.load_memory_variables({})
    
    def get_memory_as_context(self):
        # Format the conversation history as a string context
        memory_variables = self.memory.load_memory_variables({})
        return self._format_history(memory_variables["history"])
    
    def _format_history(self, history):
        # Convert the history messages to a string format for context
        formatted_history = ""
        for message in history:
            role = "User" if message.type == "human" else "Assistant"
            formatted_history += f"{role}: {message.content}\n\n"
        return formatted_history