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

draft = "our app is good it helps students manage work"
issues = [
    "Fix spelling, capitalization, and punctuation.",
    "Make the tone professional and suitable for academic software.",
    "Include a strong call-to-action to register today."
]

refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """Revise the given text to address this specific issue:
{issue_to_fix}
Keep the overall meaning intact. Return only the revised text."""),
    ("user", "Draft: {draft_text}")
])
refine_chain = refine_prompt | llm | parser

current_text = draft
for i, issue in enumerate(issues, start=1):
    current_text = refine_chain.invoke({
        "draft_text": current_text,
        "issue_to_fix": issue
    })
    print(f"Round {i} (Fixed: {issue}):\n{current_text}\n")