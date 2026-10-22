import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)
parser = StrOutputParser()

review_text = "The product stopped working after two days, very disappointed with the build."

review_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the review into exactly one label: [Positive], [Negative], or [Neutral]. Return only the label name."),
    ("user", "Customer review: {review_text}")
])

review_chain = review_prompt | llm | parser
response = review_chain.invoke({"review_text": review_text})

print(response)