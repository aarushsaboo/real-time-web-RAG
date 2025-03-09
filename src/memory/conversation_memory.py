from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

class ConversationManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def add_user_message(self, message):
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message):
        self.memory.chat_memory.add_ai_message(message)
    
    def get_chat_history(self):
        return self.memory.chat_memory.messages
    
    def get_formatted_history(self):
        # Format history as string for context
        history = ""
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                history += f"Assistant: {message.content}\n"
        return history