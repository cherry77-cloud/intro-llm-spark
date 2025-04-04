# 1. Environment setup
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://xiaoai.plus/v1"

# 2. Prepare document data
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# 3. Initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key,
    base_url=base_url
)

# 4. Create vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}  # Return the most relevant result
)

# 6. Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=openai_api_key,
    base_url=base_url
)

# 7. Create prompt template with improved instructions
template = """
Please answer the question based on the following context. If the context doesn't contain relevant information, say 
"I don't have enough information to answer this question."

Question: 
{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_template(template)

# 8. Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

if __name__ == "__main__":
    # Test vector search
    print("=== Testing Direct Search ===")
    results = vectorstore.similarity_search_with_score("cat", k=2)
    for doc, score in results:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Score: {score}")
        print("-" * 50)
    
    # Test retriever
    print("=== Testing Retriever ===")
    retriever_results = retriever.invoke("parrot")
    for doc in retriever_results:
        print(f"Retrieved result: {doc.page_content}")
        print("-" * 50)
    
    # Test complete RAG pipeline
    print("=== Testing RAG Pipeline ===")
    questions = [
        "What characteristics do cats have?",
        "What special abilities do parrots have?",
        "How do you feed sharks?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Test Question {i}: {question}")
        context_docs = retriever.invoke(question)
        print(f"Retrieved context: {context_docs[0].page_content if context_docs else 'No relevant document found'}")
        response = rag_chain.invoke(question)
        print(f"Answer: {response.content}")
        print("-" * 50)
