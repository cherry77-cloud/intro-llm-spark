import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from rich.markdown import Markdown

# Load environment variables
_ = load_dotenv(find_dotenv())

# -------------------
# 1. Load CSV data
# -------------------
file = 'clothing_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

# -------------------
# 2. Create vector store
# -------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# -------------------
# 3. Build QA chain
# -------------------
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    verbose=True
)

# -------------------
# 4. Output formatting
# -------------------
def print_markdown_table(response):
    """Pretty print Markdown table using rich"""
    from rich.console import Console
    console = Console()
    markdown_content = response["result"].strip()
    console.print(Markdown(markdown_content))


if __name__ == "__main__":
    # Optimized query for clean table output
    query = """
    List 10 shirts with sun protection in Markdown table format with:
    - Columns: Product Name | UPF Rating | Material | Price
    - Center-aligned headers (using :-: syntax)
    - Right-aligned price column (using -: syntax)
    - Remove unnecessary whitespace
    - Use consistent formatting
    """
    
    response = qa_chain.invoke(query)
    print("=== Formatted Markdown Table ===")
    print_markdown_table(response)
