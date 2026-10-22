import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Question-Answer Pairs
class QAPair(BaseModel):
    question: str
    answer: str

class TranscriptExtraction(BaseModel):
    qa_list: List[QAPair]

parser_step1 = PydanticOutputParser(pydantic_object=TranscriptExtraction)
step1_prompt = PromptTemplate(
    template="""Extract all interview questions and candidate answers from the transcript.
{format_instructions}
Transcript:
{transcript_text}""",
    input_variables=["transcript_text"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)
step1_chain = step1_prompt | llm | parser_step1

# Step 2: Skill Assessment
class SkillEvaluation(BaseModel):
    skill: str
    demonstrated: bool
    evidence_or_gap: str

class SkillEvaluationList(BaseModel):
    evaluations: List[SkillEvaluation]

parser_step2 = PydanticOutputParser(pydantic_object=SkillEvaluationList)
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical interviewer evaluating skill evidence. Evaluate each skill based on the candidate's answers and flag gaps.\n{format_instructions}"),
    ("user", "Candidate Q&A:\n{qa_text}\n\nSkills to Assess: {skills}")
]).partial(format_instructions=parser_step2.get_format_instructions())
step2_chain = step2_prompt | llm | parser_step2

# Step 3: Final Scorecard
class ScorecardRow(BaseModel):
    skill: str
    rating: str
    notes: str

class FinalScorecard(BaseModel):
    scorecard: List[ScorecardRow]
    overall_recommendation: str

parser_step3 = PydanticOutputParser(pydantic_object=FinalScorecard)
step3_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a hiring panel coordinator. Compile the skill evaluations into a final structured candidate scorecard with an overall recommendation.\n{format_instructions}"),
    ("user", "Evaluations:\n{evaluation_text}")
]).partial(format_instructions=parser_step3.get_format_instructions())
step3_chain = step3_prompt | llm | parser_step3

if __name__ == "__main__":
    transcript = """
Interviewer: How do you handle schema migrations in SQL?
Candidate: I use Alembic and ensure backward-compatible migrations with rollback plans.
Interviewer: Have you implemented distributed caching with Redis?
Candidate: No, I haven't worked with Redis or caching clusters yet.
"""
    skills = "SQL, Redis caching"

    step1_out = step1_chain.invoke({"transcript_text": transcript})
    print("--- Step 1: Extracted Q&A ---")
    print(step1_out.model_dump())

    qa_summary = "\n".join([f"Q: {i.question}\nA: {i.answer}" for i in step1_out.qa_list])
    step2_out = step2_chain.invoke({"qa_text": qa_summary, "skills": skills})
    print("\n--- Step 2: Skill Evaluations ---")
    print(step2_out.model_dump())

    eval_summary = "\n".join([f"Skill: {e.skill}, Demonstrated: {e.demonstrated}, Details: {e.evidence_or_gap}" for e in step2_out.evaluations])
    step3_out = step3_chain.invoke({"evaluation_text": eval_summary})
    print("\n--- Step 3: Final Scorecard ---")
    print(step3_out.model_dump())