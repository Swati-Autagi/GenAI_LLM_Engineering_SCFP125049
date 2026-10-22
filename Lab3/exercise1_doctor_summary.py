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

patient_name = "Swati Autagi"
patient_notes = """
Patient has persistent dry cough and mild fever 100F for 3 days.
Doctor advised saline gargles, paracetamol, and ample fluids.
"""

doctor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical summary assistant.
Use only the given patient notes. Do not invent medical information.
If diagnosis is missing, write Not provided.
Return strictly these three sections in order:
Symptoms:
Diagnosis:
Recommendation:"""),
    ("user", "Patient Name: {patient_name}\nPatient Notes: {patient_notes}")
])

doctor_chain = doctor_prompt | llm | parser
response = doctor_chain.invoke({
    "patient_name": patient_name,
    "patient_notes": patient_notes
})

print(response)