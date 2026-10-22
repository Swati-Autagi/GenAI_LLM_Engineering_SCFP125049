# Activity 5: Run an LLM Locally Using Ollama

### 1. Does the model need internet after downloading?
- No. Once the model weights and Modelfile (e.g., `ollama run llama3.2:1b`) are downloaded to local storage, Ollama functions completely offline with zero internet access.

### 2. Where is inference actually happening?
- Inference happens entirely on your local hardware: inside your laptop's CPU cores (using SIMD/AVX instructions) and/or local GPU/VRAM (via CUDA/ROCm/Metal compute shaders).

### 3. Is any data leaving your machine?
- No. All prompt tokens, internal activations, and output responses stay inside local RAM and storage. No telemetry, logs, or prompt data are transmitted to external servers.

### 4. What happens if the model is too large for your memory?
- Ollama automatically performs layer-offloading:
  * As many layers as possible are loaded into GPU VRAM.
  * Remaining layers fall back to system RAM (CPU execution).
  * If the model exceeds total combined RAM + VRAM, the operating system triggers virtual memory page swapping (disk thrashing), slowing generation to fractions of a token per second, or the process terminates with an `Out Of Memory` (OOM) error.