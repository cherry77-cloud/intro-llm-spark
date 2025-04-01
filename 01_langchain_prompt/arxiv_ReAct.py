import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentExecutor
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

# ======================
# 1. 工具组件初始化
# ======================
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper()  # Arxiv API封装器
)

tools = [
    Tool(
        name="Search",
        func=arxiv_tool.run,
        description="Search for academic papers on ArXiv."
    ),
    Tool(
        name="Lookup",
        func=arxiv_tool.run,
        description="Lookup detailed information about a paper."
    )
]

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

# ======================
# 2. 智能体构建
# ======================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True 
)

# ======================
# 3. 执行示例
# ======================
if __name__ == "__main__":
    question = "What are recent papers about Large Language Models?"
    result = agent.invoke({"input": question})
    print(result["output"])
