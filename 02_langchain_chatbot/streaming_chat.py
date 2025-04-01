import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


_ = load_dotenv(find_dotenv())
model = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1",
    streaming=True
)

trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=model,
    include_system=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 核心处理链（包含裁剪和流式支持）
chain = (
    RunnablePassthrough.assign(
        chat_history=itemgetter("chat_history") | trimmer
    )
    | prompt
    | model
    | StrOutputParser()
)


if __name__ == "__main__":
    history = [
        HumanMessage(content="Hi there!"),
        AIMessage(content="Hello! How can I help?"),
        HumanMessage(content="What's 2+2?"),
        AIMessage(content="The answer is 4")
    ]

    print("--- Regular Call ---")
    response = chain.invoke({
        "input": "What was my last math question?",
        "chat_history": history
    })
    print(response)

    print("--- Streaming Call ---")
    for chunk in chain.stream({
        "input": "Explain the solution step by step",
        "chat_history": history + [
            HumanMessage(content="What was my last math question?"),
            AIMessage(content="You asked: What's 2+2?")
        ]
    }):
        print(chunk, end="", flush=True)
