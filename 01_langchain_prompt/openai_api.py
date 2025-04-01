import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 初始化LangChain的ChatOpenAI客户端
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

def get_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

# 测试调用
if __name__ == "__main__":
    print(get_completion("What is 1+1?"))
