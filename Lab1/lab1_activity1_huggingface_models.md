# Activity 1: Explore LLM Models on Hugging Face

### 1. Small Model: Qwen/Qwen2.5-0.5B-Instruct
- **Creator:** Alibaba Cloud (Qwen Team)
- **Parameters:** ~0.49 Billion (~490 Million)
- **Generative Model:** Yes, autoregressive causal language model.
- **Context Length:** 32,768 tokens (supports up to 128k with rope scaling).
- **Suitable for Local Execution:** Yes, runs effortlessly on low-end laptops, mobile devices, and even without a dedicated GPU.
- **Recommended Hardware:** Standard CPU with 2 GB RAM or an entry-level GPU with 2 GB VRAM.

### 2. Medium Model: meta-llama/Llama-3.1-8B-Instruct
- **Creator:** Meta AI
- **Parameters:** 8.03 Billion
- **Generative Model:** Yes, text-to-text autoregressive generative model.
- **Context Length:** 128,000 tokens (128k).
- **Suitable for Local Execution:** Yes, easily runs locally when quantized (e.g., 4-bit INT4 requires ~5.5 GB to 6 GB VRAM).
- **Recommended Hardware:** GPU with at least 8 GB to 16 GB VRAM (e.g., RTX 3060/4060) or 16 GB system RAM for CPU offloading.

### 3. Large Model: meta-llama/Llama-3.1-70B-Instruct
- **Creator:** Meta AI
- **Parameters:** 70.6 Billion
- **Generative Model:** Yes, instruction-tuned generative model.
- **Context Length:** 128,000 tokens.
- **Suitable for Local Execution:** Difficult on personal laptops. Requires multi-GPU setups or high-end Mac Apple Silicon (64 GB+ unified memory) to run quantized.
- **Recommended Hardware:** 2x to 4x 24GB GPUs (e.g., dual RTX 3090/4090 or A100) or 64 GB - 128 GB system RAM for INT4 inference.