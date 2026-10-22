import tiktoken

# Load OpenAI/Groq standard tokenizer (cl100k_base)
enc = tiktoken.get_encoding("cl100k_base")

def analyze_tokenization(text: str):
    token_ids = enc.encode(text)
    token_strings = [enc.decode([tid]) for tid in token_ids]
    print(f"\nOriginal Text: \"{text}\"")
    print(f"Token Count: {len(token_ids)}")
    print(f"Tokens: {token_strings}")
    print(f"Token IDs: {token_ids}")

if __name__ == "__main__":
    print("--- Activity 2: Tokenization Practical ---")
    analyze_tokenization("Learning Generative AI is exciting!")
    analyze_tokenization("antidisestablishmentarianism")
    analyze_tokenization("Hello, world! How's it going?")
    analyze_tokenization("1234567890")

    print("""
======================================================
Answers to Activity 2 Questions:
======================================================
1. Is one word always one token?
   - No. Common short words (e.g., "is", "AI") are often single tokens, but rare, complex, or long words are split into multiple sub-word tokens.

2. Why are some words split?
   - Tokenizers use sub-word algorithms like Byte-Pair Encoding (BPE). Splitting rare or compound words (e.g., "antidisestablishmentarianism") allows the model to handle an infinite vocabulary using a finite vocabulary dictionary (~100k tokens) without running into Out-Of-Vocabulary (OOV) errors.

3. What happens to punctuation?
   - Punctuation marks (like commas, periods, exclamation marks) are either encoded as individual tokens or grouped together with preceding/following whitespace.

4. Why does the model need token IDs?
   - Neural networks cannot perform mathematical operations on raw text strings. Token IDs convert text into numerical integers that index directly into the model's high-dimensional vector embedding space.
======================================================
""")