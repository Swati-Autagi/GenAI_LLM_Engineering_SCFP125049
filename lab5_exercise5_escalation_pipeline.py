import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Structured Chat Summary
class ChatSummary(BaseModel):
    issue: str
    customer_tone: str
    resolution_attempted: str

parser_step1 = PydanticOutputParser(pydantic_object=ChatSummary)
step1_prompt = PromptTemplate(
    template="""Summarize the messy customer support chat log into structured fields.
{format_instructions}
Chat Log:
{chat_log_text}""",
    input_variables=["chat_log_text"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)
step1_chain = step1_prompt | llm | parser_step1

# Step 2: Gap / Readiness Check
class ReadinessCheck(BaseModel):
    missing_fields: List[str] = Field(description="List of missing crucial escalation fields like account ID, error code, invoice ID")
    is_ready_to_escalate: bool = Field(description="True if all critical details are present, otherwise False")

parser_step2 = PydanticOutputParser(pydantic_object=ReadinessCheck)
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an escalation-readiness reviewer. Check if all required info (Account ID, Error Code, Device Info) is present before escalating.\n{format_instructions}"),
    ("user", "Issue: {issue}\nTone: {customer_tone}\nResolution Attempted: {resolution_attempted}")
]).partial(format_instructions=parser_step2.get_format_instructions())
step2_chain = step2_prompt | llm | parser_step2

# Step 3: Prioritized Escalation Report
step3_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support operations lead. Write a prioritized escalation report for senior support engineers based on the readiness review."),
    ("user", """Structured Incident:
- Issue: {issue}
- Tone: {customer_tone}
- Resolution Attempted: {resolution_attempted}
- Missing Information: {missing_fields}
- Escalation Ready: {is_ready_to_escalate}

Generate the final escalation report.""")
])
step3_chain = step3_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    chat_sample = """
User: Hey, my transaction failed with Error 502, but money was deducted!
Agent: I am checking. Can you provide your registered email?
User: swati@example.com. Please resolve this immediately, this is urgent!
Agent: I tried refreshing the payment webhook, but it still shows pending.
"""
    s1_out = step1_chain.invoke({"chat_log_text": chat_sample})
    print("--- Step 1: Structured Summary ---")
    print(s1_out.model_dump())

    s2_out = step2_chain.invoke(s1_out.model_dump())
    print("\n--- Step 2: Escalation Readiness Check ---")
    print(s2_out.model_dump())

    merged_data = {**s1_out.model_dump(), **s2_out.model_dump()}
    s3_out = step3_chain.invoke(merged_data)
    print("\n--- Step 3: Prioritized Escalation Report ---")
    print(s3_out)