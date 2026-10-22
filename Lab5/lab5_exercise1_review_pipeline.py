import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Schema & Extraction (PromptTemplate + PydanticOutputParser)
class ReviewExtraction(BaseModel):
    complaint: str = Field(description="The core customer complaint")
    product_or_feature: str = Field(description="The specific product or feature mentioned")
    sentiment: str = Field(description="Customer sentiment (e.g., negative, frustrated)")

parser_step1 = PydanticOutputParser(pydantic_object=ReviewExtraction)

step1_prompt = PromptTemplate(
    template="""Extract the core complaint, product/feature mentioned, and customer sentiment from this review.
{format_instructions}
Customer Review:
{review_text}""",
    input_variables=["review_text"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)

step1_chain = step1_prompt | llm | parser_step1

# Step 2: Generation via Persona (ChatPromptTemplate + StrOutputParser)
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an internal support ticket writer. Create an internal support ticket summary using ONLY the provided structured details."),
    ("user", """Structured Review Details:
Complaint: {complaint}
Product/Feature: {product_or_feature}
Sentiment: {sentiment}

Generate a concise internal support ticket summary.""")
])

step2_chain = step2_prompt | llm | StrOutputParser()

# Execution
review_input = "The battery of the wireless mouse dies in less than 4 hours. It's completely unacceptable for an expensive device!"

extracted_data = step1_chain.invoke({"review_text": review_input})
print("--- Step 1: Structured Extraction ---")
print(extracted_data.model_dump())

ticket_summary = step2_chain.invoke(extracted_data.model_dump())
print("\n--- Step 2: Support Ticket Summary ---")
print(ticket_summary)