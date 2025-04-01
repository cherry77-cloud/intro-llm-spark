import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class SessionManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance.sessions = {}
        return cls._instance
    
    def get_session_history(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]
    
    def get_all_sessions(self):
        return self.sessions

# 初始化会话管理器
session_manager = SessionManager()

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(
    model_name="gpt-4-turbo-2024-04-09",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
chat_model = RunnableWithMessageHistory(
    chain,
    session_manager.get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

response = chat_model.invoke(
    {"input": "Hi! I'm Alice"},
    config={"configurable": {"session_id": "user_a"}}
)
print("user: Hi! I'm Alice")
print(f"AI: {response.content}")

# 同一会话记住上下文
response = chat_model.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user_a"}}
)
print("user: What's my name?")
print(f"AI: {response.content}")

# 完全隔离
response = chat_model.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user_b"}}
)
print("user: What's my name?")
print(f"AI: {response.content}")
