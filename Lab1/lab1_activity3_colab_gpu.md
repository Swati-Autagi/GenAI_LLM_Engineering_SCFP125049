# Activity 3: Run an LLM Using Google Colab GPU

### 1. Is the GPU actually being used?
- Yes. Inspecting `nvidia-smi` in Google Colab (or running `torch.cuda.is_available()`) shows the GPU device active (e.g., NVIDIA Tesla T4 15-16 GB). During model generation, GPU Core Utilization rises to 70%–100%, and VRAM is allocated for model weights and CUDA tensors.

### 2. How much memory does the model consume?
- A small model like `Qwen/Qwen2.5-0.5B` consumes ~1.5 GB to 2 GB VRAM.
- A 7B/8B model loaded in FP16 consumes ~15 GB to 16 GB VRAM, fitting right at the limit of a standard Colab free-tier T4 GPU. Loaded in 4-bit (bitsandbytes), it consumes ~5.5 GB to 6 GB VRAM.

### 3. What happens if you increase the output length?
- As `max_new_tokens` increases, inference latency increases linearly because autoregressive generation generates one token at a time.
- Memory consumption rises due to the growing KV (Key-Value) cache, which stores past attention states for all generated tokens in GPU memory.

### 4. Would this run on a laptop without a GPU?
- Small models (0.5B - 3B) can run on a standard laptop CPU using libraries like `llama.cpp` or quantized Ollama.
- However, generating long responses on a laptop CPU without AVX-512 or dedicated GPU tensor cores is significantly slower (e.g., 2–5 tokens/sec compared to 40–80 tokens/sec on a GPU).