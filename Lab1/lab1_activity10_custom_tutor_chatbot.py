import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

def run_tutor_simulation():
    # Role-based system instruction
    chat_history = [
        SystemMessage(content="""You are an encouraging AI/ML Professor.
Rules:
1. Always explain machine learning concepts using simple real-world analogies.
2. Ask one checking question at the end to confirm understanding.
3. Be enthusiastic and academic in tone.""")
    ]
    
    user_query = "Can you explain what Overfitting means?"
    print("--- Activity 10: Customized Role-Based Chatbot ---")
    print(f"User: {user_query}")
    
    chat_history.append(HumanMessage(content=user_query))
    response = llm.invoke(chat_history)
    print(f"\nAI Tutor:\n{response.content}")

if __name__ == "__main__":
    run_tutor_simulation()

    print("""
======================================================
Answers to Activity 10 Questions:
======================================================
1. How did the responses change compared to Activity 9?
   - In Activity 9, the assistant answered plainly and neutrally. With the custom role prompt, the assistant adopted an encouraging pedagogical tone, used an intuitive analogy, and concluded with a follow-up test question.

2. Could the same model behave like a completely different assistant just from instructions?
   - Yes. System prompts set the persona, tone, guardrails, and behavioral parameters of the LLM. The underlying model weights remain identical, but its conditional token generation changes completely based on the system prompt context.

3. What role would you build next, and why?
   - An "Automated Code Reviewer and Debugging Coach" that analyzes students' Python scripts, explains the time/space complexity, and highlights anti-patterns without giving away direct answers, fostering independent problem-solving skills.
======================================================
""")