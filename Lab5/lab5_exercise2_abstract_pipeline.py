import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Extraction Schema
class AbstractExtraction(BaseModel):
    research_question: str = Field(description="The main research question investigated")
    method: str = Field(description="The scientific method or technique employed")
    key_finding: str = Field(description="The core finding or result")

parser_step1 = PydanticOutputParser(pydantic_object=AbstractExtraction)

step1_prompt = PromptTemplate(
    template="""Extract the research question, method, and key finding from the abstract.
{format_instructions}
Abstract:
{abstract_text}""",
    input_variables=["abstract_text"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)

step1_chain = step1_prompt | llm | parser_step1

# Step 2: Layperson Summary
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a science communicator explaining complex research to a general audience. Use simple words and avoid technical jargon."),
    ("user", """Research Details:
Question: {research_question}
Method: {method}
Finding: {key_finding}

Write a plain-language summary for the public.""")
])

step2_chain = step2_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    abstract_sample = """This study explores whether daily 15-minute mindfulness meditation reduces self-reported stress levels in college students. We conducted a randomized controlled trial with 250 participants over 8 weeks, monitoring salivary cortisol levels and psychological stress surveys. The intervention group demonstrated a statistically significant 23% reduction in cortisol compared to the baseline and control group."""
    
    extracted_data = step1_chain.invoke({"abstract_text": abstract_sample})
    print("--- Step 1: Structured Abstract Extraction ---")
    print(extracted_data.model_dump())

    layperson_summary = step2_chain.invoke(extracted_data.model_dump())
    print("\n--- Step 2: Layperson Summary ---")
    print(layperson_summary)