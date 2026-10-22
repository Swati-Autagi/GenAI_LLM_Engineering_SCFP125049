import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Goal Extraction
class FitnessGoalDetails(BaseModel):
    goal_type: str
    fitness_level: str
    days_available: int
    equipment_access: str

parser_step1 = PydanticOutputParser(pydantic_object=FitnessGoalDetails)
step1_prompt = PromptTemplate(
    template="""Extract the user's fitness goal details.
{format_instructions}
User Goal: {fitness_goal_text}
Available Days: {days_per_week}""",
    input_variables=["fitness_goal_text", "days_per_week"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)
step1_chain = step1_prompt | llm | parser_step1

# Step 2: Day-by-Day Plan
class DailyWorkout(BaseModel):
    day: str
    focus: str
    exercises: List[str]

class WeeklyWorkoutPlan(BaseModel):
    plan: List[DailyWorkout]

parser_step2 = PydanticOutputParser(pydantic_object=WeeklyWorkoutPlan)
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a certified personal trainer. Generate a structured day-by-day weekly workout plan matching the user's availability and equipment.\n{format_instructions}"),
    ("user", """Client Details:
- Goal: {goal_type}
- Current Level: {fitness_level}
- Days per Week: {days_available}
- Equipment: {equipment_access}""")
]).partial(format_instructions=parser_step2.get_format_instructions())
step2_chain = step2_prompt | llm | parser_step2

if __name__ == "__main__":
    goal_text = "I am a complete beginner looking to build lean muscle. I only have dumbbells at home."
    days = 3

    step1_out = step1_chain.invoke({"fitness_goal_text": goal_text, "days_per_week": days})
    print("--- Step 1: Goal Details ---")
    print(step1_out.model_dump())

    step2_out = step2_chain.invoke(step1_out.model_dump())
    print("\n--- Step 2: Weekly Workout Plan ---")
    print(step2_out.model_dump())