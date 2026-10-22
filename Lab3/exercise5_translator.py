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

sentence = "Could you please help me solve this problem?"
target_language = "Marathi"
formality = "formal"

translation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Translate the sentence into {target_language}.
Use a {formality} formality level. Return ONLY the translated sentence."""),
    ("user", "Sentence: {sentence}")
])

translation_chain = translation_prompt | llm | parser
response = translation_chain.invoke({
    "sentence": sentence,
    "target_language": target_language,
    "formality": formality
})

print(response)