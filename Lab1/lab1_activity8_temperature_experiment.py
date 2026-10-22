import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

prompt = "Complete this sentence in creative ways: The abandoned lighthouse was known for..."

temperatures = [0.1, 0.7, 1.0]

if __name__ == "__main__":
    print("--- Activity 8: Temperature Experiment ---")
    
    for temp in temperatures:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temp
        )
        print(f"\n[Testing Temperature = {temp}]")
        for i in range(2):
            resp = llm.invoke(prompt)
            print(f"Run {i+1}: {resp.content.strip()}")

    print("""
======================================================
Answers to Activity 8 Questions:
======================================================
1. Which output felt most predictable? Most creative?
   - Temperature 0.1 was the most predictable; repeated runs produced near-identical, deterministic text following the most probable token trajectories.
   - Temperature 1.0 was the most creative and diverse, selecting less probable tokens and producing unexpected vocabulary and imagery.

2. Does temperature change what the model knows?
   - No. Temperature does not alter the model's weights, training data, or knowledge base. It only alters the Softmax probability distribution sampling when picking the next token.

3. When would you want low temperature? High temperature?
   - Low Temperature (0.0 - 0.2): Factual Q&A, SQL generation, JSON extraction, medical/legal summaries, and deterministic coding tasks.
   - High Temperature (0.7 - 1.0+): Creative writing, poetry, brainstorms, marketing slogans, and dialogue generation.
======================================================
""")