import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key=api_key,
)

def get_completion(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

print(get_completion("What is 1+1?"))
