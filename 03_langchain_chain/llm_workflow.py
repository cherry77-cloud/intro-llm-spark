import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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
translate_chain = (
    ChatPromptTemplate.from_template("Translate the following review to english:\n\n{Review}")
    | llm
    | StrOutputParser()
).with_config(name="translate_review")

summarize_chain = (
    ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:\n\n{English_Review}")
    | llm
    | StrOutputParser()
).with_config(name="generate_summary")

detect_language_chain = (
    ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
    | llm
    | StrOutputParser()
).with_config(name="detect_language")

followup_chain = (
    ChatPromptTemplate.from_template(
        "Write a follow up response to the following summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    | llm
    | StrOutputParser()
).with_config(name="generate_followup")

# 2. 组合成完整工作流
overall_chain = (
    # 先翻译评论
    RunnablePassthrough.assign(
        English_Review=translate_chain
    )
    # 然后使用翻译结果生成摘要
    | RunnablePassthrough.assign(
        summary=summarize_chain
    )
    # 最后并行检测语言和生成跟进消息
    | RunnableParallel(
        English_Review=lambda x: x["English_Review"],  # 保留翻译结果
        summary=lambda x: x["summary"],  # 保留摘要结果
        detect_language=RunnablePassthrough.assign(
            Review=lambda x: x["Review"]
        ) | detect_language_chain,
        followup_message=lambda x: followup_chain.invoke({
            "summary": x["summary"],
            "language": detect_language_chain.invoke({"Review": x["Review"]})
        })
    )
)

# 3. 执行示例
review = "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre..."
result = overall_chain.invoke({"Review": review})

# 4. 格式化输出
final_result = {
    "English_Review": result["English_Review"],
    "summary": result["summary"],
    "language": result["detect_language"],
    "followup_message": result["followup_message"]
}
print(final_result)
