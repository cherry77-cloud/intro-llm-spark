import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize model
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

# ========== 1. Define domain templates ==========
DOMAIN_TEMPLATES = {
    "physics": """You are a physics professor, please answer this physics question: {input}""",
    "math": """You are a mathematician, please solve this math problem: {input}""",
    "history": """You are a historian, please analyze this history question: {input}""",
    "cs": """You are a computer scientist, please explain this technical question: {input}""",
    "default": """You are an AI assistant, please answer: {input}"""
}

# ========== 2. Build processing chains ==========
def build_chain(prompt_template: str, domain: str):
    """Build domain processing chain"""
    return (
        ChatPromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    ).with_config(name=f"{domain}_chain")

chains = {
    domain: build_chain(template, domain)
    for domain, template in DOMAIN_TEMPLATES.items()
}

# ========== 3. Routing system ==========
def route_with_logging(input_data: dict) -> dict:
    """Routing function with logging"""
    question = input_data["input"]
    
    # Display routing process
    print(f"\n🔍 Analyzing route... (question: {question})")
    
    # Get routing decision
    domain = (
        ChatPromptTemplate.from_template("""
        Please determine the most suitable domain for this question (physics/math/history/cs/default):
        Question: {input}
        Return only the domain name, no explanation.""")
        | llm
        | StrOutputParser()
    ).invoke({"input": question})
    
    print(f"✅ Routing decision: [ {domain} ]")
    return {"domain": domain, "input": question}

# ========== 4. Build main chain ==========
main_chain = (
    RunnablePassthrough()
    | RunnableLambda(route_with_logging)  # Routing layer with logging
    | {
        "answer": RunnableBranch(
            (lambda x: x["domain"] == "physics", chains["physics"]),
            (lambda x: x["domain"] == "math", chains["math"]),
            (lambda x: x["domain"] == "history", chains["history"]),
            (lambda x: x["domain"] == "cs", chains["cs"]),
            chains["default"]
        ),
        "domain": lambda x: x["domain"],  # Preserve routing information
        "question": lambda x: x["input"]  # Preserve original question
    }
)

# ========== 5. Run examples ==========
if __name__ == "__main__":
    questions = [
        "Explain quantum entanglement",
        "Find the roots of x² + 5x + 6 = 0",
        "What is the historical significance of Zheng He's voyages?",
        "What are the design principles of RESTful API?",
        "How to make scrambled eggs with tomatoes?"
    ]
    
    for q in questions:
        print(f"\n{'='*50}")
        print(f"📌 Original question: {q}")
        result = main_chain.invoke({"input": q})
        print(f"\n🏷️ Routed to: {result['domain']}")
        print(f"💡 Answer: {result['answer']}")
        print("="*50)
