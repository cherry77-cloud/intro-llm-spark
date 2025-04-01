import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

_ = load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']
base_url = 'https://xiaoai.plus/v1'

model = ChatOpenAI(
    model="gpt-4-turbo",
    base_url=base_url,
    api_key=api_key
)

messages = [
    SystemMessage(content="Translate English to Italian"),
    HumanMessage(content="hi")
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate into {language}:"),
    ("user", "{text}")
])

# 创建处理链：提示模板 -> 模型 -> 字符串输出解析器
chain = prompt | model | StrOutputParser()
result = chain.invoke({"language": "italian", "text": "hi"})
