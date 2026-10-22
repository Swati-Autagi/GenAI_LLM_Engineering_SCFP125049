# Activity 4: Can My Computer Run an LLM?

### 1. Your laptop has 8 GB RAM — 4B FP16, 4B INT4, or 7B INT4? Why?
- **Recommendation: 4B INT4.**
- **Calculation & Breakdown:**
  * **4B FP16:** Model weights alone require `4B * 2 bytes = 8 GB`. The OS and system processes alone require 2.5–3.5 GB of RAM, causing an instant Out-Of-Memory (OOM) crash or extreme disk swapping.
  * **7B INT4:** Weights require `7B * 0.5 bytes = 3.5 GB`. Adding runtime context, KV cache, and OS background tasks consumes 7.5–8.2 GB, causing severe lag and thrashing on an 8 GB system.
  * **4B INT4:** Weights require `4B * 0.5 bytes = 2.0 GB`. With runtime buffers (1 GB) and OS memory (3 GB), total usage is ~6 GB, leaving a comfortable 2 GB buffer for stable execution.

### 2. What else besides model weights uses memory?
1. **Operating System & Background Processes:** Windows/macOS consumes 2.5 GB to 4 GB RAM.
2. **KV Cache (Key-Value Cache):** Stores attention vectors for input and generated tokens. Scales with context length and batch size.
3. **Activation Memory:** Intermediate tensor outputs computed during the forward pass.
4. **CUDA Runtime Overhead:** Memory reserved by GPU drivers and CUDA context.

### 3. Would quantization always be the safer choice? Why or why not?
- **Safer for Memory:** Yes, quantization (FP16 → INT8 → INT4) drops memory usage by 50% to 75% and enables models to run on consumer hardware.
- **Trade-offs:** 
  * Aggressive quantization (e.g., INT2 or extreme INT3) leads to perplexity degradation, hallucinations, and loss of mathematical/reasoning accuracy.
  * Dequantization operations can slightly increase compute overhead on older CPUs lacking INT4 instruction acceleration.