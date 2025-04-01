import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# ======================
# 1. 环境初始化
# ======================
_ = load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

# ======================
# 2. 核心组件初始化
# ======================
model = ChatOpenAI(
    model="gpt-4-turbo",
    base_url='https://xiaoai.plus/v1',
    api_key=api_key,
    temperature=0
)

messages = [
    SystemMessage(content="Translate English to Italian"),
    HumanMessage(content="hi")
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate into {language}:"),  # 支持动态语言参数
    ("user", "{text}")                         # 用户文本输入插槽
])

# ======================
# 3. 处理链构建
# ======================
# 使用LCEL管道运算符组合组件：
# prompt模板 -> 模型调用 -> 字符串输出解析
chain = prompt | model | StrOutputParser()

# ======================
# 4. 调用演示
# ======================
if __name__ == "__main__":
    direct_response = model.invoke(messages)
    print(direct_response.content)  # 输出: "ciao"
    
    # 链式调用演示
    chain_response = chain.invoke({
        "language": "italian",  # 动态参数注入
        "text": "hi"            # 用户输入文本
    })
    print(chain_response)       # 输出: "ciao"
