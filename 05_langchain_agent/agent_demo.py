import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# 1. Initialize Components
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

search_tool = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY,
    max_results=2
)


# 2. Create Agent with Memory
agent_executor = create_react_agent(
    model=llm.bind_tools([search_tool]),
    tools=[search_tool],
    checkpointer=MemorySaver()
).with_config({"run_name": "Agent"})


# 3. Synchronous Query
def sync_query():
    print("[SYNC QUERY]")
    config = {"configurable": {"thread_id": "sync-1"}}
    response = agent_executor.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco?")]
    }, config)
    print("Final Answer:", response["messages"][-1].content)


# 4. Memory Conversation
def memory_conversation():
    print("[MEMORY CONVERSATION]")
    config = {"configurable": {"thread_id": "conv-1"}}
    
    # First message
    agent_executor.invoke({
        "messages": [HumanMessage(content="My name is John Doe")]
    }, config)
    
    # Second message
    response = agent_executor.invoke({
        "messages": [HumanMessage(content="What's my name?")]
    }, config)
    print("Name Response:", response["messages"][-1].content)


# Main Execution
if __name__ == "__main__":
    sync_query()
    memory_conversation()
