import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)
json_parser = JsonOutputParser()

resume_text = """
Name: Swati Autagi
Skills: Python, SQL, Machine Learning
Experience: 1 year data analytics
Email: swati@example.com
"""
fields_to_extract = ["name", "skills", "email"]

resume_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract only the requested fields from the resume.
Return valid JSON only without Markdown tags.
Use null for missing fields. Do not guess.
{format_instructions}"""),
    ("user", "Resume: {resume_text}\nFields: {fields_to_extract}")
]).partial(format_instructions=json_parser.get_format_instructions())

resume_chain = resume_prompt | llm | json_parser
response = resume_chain.invoke({
    "resume_text": resume_text,
    "fields_to_extract": fields_to_extract
})

print(response)