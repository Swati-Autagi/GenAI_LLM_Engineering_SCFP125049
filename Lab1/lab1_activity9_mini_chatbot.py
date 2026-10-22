import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

def run_chat_simulation():
    chat_history = [
        SystemMessage(content="You are a helpful and concise AI assistant.")
    ]
    
    turns = [
        "Hi, my name is Swati Autagi.",
        "What is my name?",
        "What degree am I studying for?"
    ]

    print("--- Activity 9: Multi-Turn Mini Chatbot Simulation ---")
    for user_text in turns:
        print(f"\nUser: {user_text}")
        chat_history.append(HumanMessage(content=user_text))
        response = llm.invoke(chat_history)
        print(f"Assistant: {response.content}")
        chat_history.append(AIMessage(content=response.content))

if __name__ == "__main__":
    run_chat_simulation()

    print("""
======================================================
Answers to Activity 9 Questions:
======================================================
1. Why does the chatbot need to remember previous messages?
   - LLMs are completely stateless functions: each API call is independent. To maintain context and handle follow-up pronouns (e.g., "What is *my* name?"), the application must explicitly resend the conversation history with each turn.

2. What happens if you drop the conversation history?
   - The model forgets all context from previous turns. Asking "What is my name?" in a fresh request without history results in the model replying that it does not know your name.

3. What's the difference between this and a single one-off prompt?
   - A single one-off prompt treats the query in total isolation. A conversational loop appends and manages an evolving dialogue state across multiple user interactions.
======================================================
""")