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

customer_message = "I received a damaged package and need a replacement immediately."
company_name = "QuickMart"
max_words = 30

support_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent for {company_name}.
Write an empathetic and professional reply.
Constraint: Your response MUST NOT exceed {max_words} words."""),
    ("user", "Customer Message: {customer_message}")
])

support_chain = support_prompt | llm | parser
response = support_chain.invoke({
    "customer_message": customer_message,
    "company_name": company_name,
    "max_words": max_words
})

print(response)