import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 初始化模型
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

# 第一步：生成公司名称
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
chain_one = first_prompt | llm | StrOutputParser()

# 第二步：生成公司描述
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}"
)
chain_two = second_prompt | llm | StrOutputParser()

# 组合成顺序链
overall_chain = RunnableSequence(
    RunnablePassthrough.assign(company_name=chain_one),
    RunnablePassthrough.assign(description=chain_two)
)

# 执行示例
product = "Queen Size Sheet Set"
result = overall_chain.invoke({"product": product})
print("Company Name:", result["company_name"])
print("Description:", result["description"])
