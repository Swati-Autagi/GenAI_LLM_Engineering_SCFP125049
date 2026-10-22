import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

if __name__ == "__main__":
    prompt = "Explain in two sentences what happens under the hood when calling an LLM API."
    
    start_time = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start_time

    print("--- Activity 6: Inference via API Provider ---")
    print(f"Response:\n{response.content}\n")
    print(f"Roundtrip Latency: {elapsed:.2f} seconds")

    print("""
======================================================
Answers to Activity 6 Questions:
======================================================
1. What is actually happening between your program and the model?
   - Your local application serializes the prompt into an HTTP POST request payload (JSON) and sends it over TLS/HTTPS to the cloud provider's API gateway.
   - The provider routes the payload to an inference cluster equipped with high-end accelerator hardware (e.g., Groq LPUs or NVIDIA H100s).
   - The server tokenizes the input, executes forward passes, decodes the output tokens, and streams/returns the final text in an HTTP response.

2. Who is paying for compute here?
   - The API provider (e.g., Groq, OpenAI, Google) operates and powers the multi-million-dollar server racks and electricity.
   - The developer/user pays the provider either via pay-as-you-go pricing (cost per million input/output tokens) or accesses a sponsored free tier.

3. What are the trade-offs versus running locally?
   - API Pros: Zero local hardware requirement, blazing fast speed on massive models (70B+), instant setup.
   - API Cons: Requires continuous internet access, costs scale with token volume, vendor lock-in, and data privacy considerations (prompts travel to third-party servers).
======================================================
""")