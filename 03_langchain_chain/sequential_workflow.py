import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

# 1. 构建所有子链
translate_review = (
    ChatPromptTemplate.from_template("Translate to english:\n\n{Review}")
    | llm
    | StrOutputParser()
)

generate_summary = (
    ChatPromptTemplate.from_template("Summarize in 1 sentence:\n\n{English_Review}")
    | llm
    | StrOutputParser()
)

detect_language = (
    ChatPromptTemplate.from_template("Detect language of:\n\n{Review}")
    | llm
    | StrOutputParser()
)

generate_followup = (
    ChatPromptTemplate.from_template(
        "Write followup response in {language} for:\n\nSummary: {summary}"
    )
    | llm
    | StrOutputParser()
)

sequential_flow = (
    RunnablePassthrough.assign(English_Review=translate_review)
    | RunnablePassthrough.assign(summary=generate_summary)
    | RunnablePassthrough.assign(language=lambda x: detect_language.invoke({"Review": x["Review"]}))
    | RunnablePassthrough.assign(
        followup_message=lambda x: generate_followup.invoke({
            "summary": x["summary"],
            "language": x["language"]
        })
    )
)


if __name__ == "__main__":
    french_review = (
        "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. "
        "J'achète les mêmes dans le commerce et le goût est bien meilleur..."
    )
    
    result = sequential_flow.invoke({"Review": french_review})
    final_result = {
        "English_Review": result["English_Review"],
        "summary": result["summary"],
        "language": result["language"],
        "followup_message": result["followup_message"]
    }
    print(final_result)
