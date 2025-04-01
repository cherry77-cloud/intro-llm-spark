import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load env
_ = load_dotenv(find_dotenv())

# Initialize
llm = ChatOpenAI(
    model_name="gpt-4-turbo-2024-04-09",
    temperature=0,
    base_url="https://xiaoai.plus/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Demo
conversation.invoke({"input": "Hi, I'm Andrew"})
conversation.invoke({"input": "What's 1+1?"})
response = conversation.invoke({"input": "What's my name?"})
print("[Buffer Memory] Final Response:", response["response"])
print("[Buffer Memory] History:", memory.buffer)
