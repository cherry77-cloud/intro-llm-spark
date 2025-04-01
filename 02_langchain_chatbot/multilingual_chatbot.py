import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 初始化LLM
_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://xiaoai.plus/v1"
)

# 全局会话存储
session_store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 构建对话链
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond in {language}:"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
chat_agent = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def run_conversation(session_id: str, language: str) -> None:
    # 第一轮对话
    chat_agent.invoke(
        {"input": "Hi I'm Sam", "language": language},
        config={"configurable": {"session_id": session_id}}
    )
    
    # 第二轮对话
    chat_agent.invoke(
        {"input": "What's my name?", "language": language},
        config={"configurable": {"session_id": session_id}}
    )

def print_session_history(session_id: str):
    print(f"Conversation history (Session: {session_id})")
    history = get_session_history(session_id)
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "AI Assistant"
        print(f"[{role}]: {msg.content}")


if __name__ == "__main__":
    run_conversation("session_1", "English")
    run_conversation("session_2", "Spanish")
    
    print_session_history("session_1")
    print_session_history("session_2")
