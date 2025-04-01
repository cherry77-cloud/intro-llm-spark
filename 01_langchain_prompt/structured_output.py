import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnablePassthrough

# ----------------------------
# 1. 初始化所有独立组件
# ----------------------------

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 定义响应模式
response_schemas = [
    ResponseSchema(
        name="gift",
        description="Was the item purchased as a gift? True/False"
    ),
    ResponseSchema(
        name="delivery_days",
        description="Delivery time in days. -1 if unknown"
    ),
    ResponseSchema(
        name="price_value",
        description="Price/value mentions as Python list"
    )
]

# 初始化输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 初始化提示模板
prompt_template = ChatPromptTemplate.from_template("""
    For the following text, extract:
    gift: Was it a gift? (True/False)
    delivery_days: Delivery days (-1 if unknown)
    price_value: Price mentions as list

    text: {text}
    {format_instructions}""")

# 初始化LLM模型
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

# ----------------------------
# 2. 组装LCEL链条
# ----------------------------

# 输入处理节点
input_processor = {
    "text": RunnablePassthrough(),
    "format_instructions": lambda _: format_instructions
}

# 完整处理链
review_chain = (
    input_processor
    | prompt_template
    | llm
    | output_parser
)

# ----------------------------
# 3. 使用示例
# ----------------------------

if __name__ == "__main__":
    customer_review = """
    This leaf blower arrived in two days, just in time for my wife's anniversary present. 
    It's slightly more expensive than others but worth it for the features."""
    
    output_dict = review_chain.invoke(customer_review)
    print(output_dict)
