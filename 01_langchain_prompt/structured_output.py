import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

_ = load_dotenv(find_dotenv())
chat = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

review_template = """
For the following text, extract the following information:

gift: Was the item purchased as a gift? Answer True/False.
delivery_days: Delivery time in days. If not found, output -1.
price_value: Extract price/value mentions as comma-separated list.

text: {text}
{format_instructions}"""

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

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template(template=review_template)

customer_review = """
This leaf blower arrived in two days, just in time for my wife's anniversary present. 
It's slightly more expensive than others but worth it for the features."""

messages = prompt.format_messages(
    text=customer_review,
    format_instructions=format_instructions
)

response = chat.invoke(messages)
output_dict = output_parser.parse(response.content)
