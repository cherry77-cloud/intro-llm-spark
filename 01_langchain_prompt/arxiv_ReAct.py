import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentExecutor
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
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
    model_name="gpt-4-turbo-2024-04-09",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

question = "What are recent papers about Large Language Models?"
result = agent.invoke({"input": question})
