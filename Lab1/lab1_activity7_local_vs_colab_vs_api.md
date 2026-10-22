# Activity 7: Compare Local vs Colab vs API

### Comparison Matrix

| Feature | Local Inference (Ollama/llama.cpp) | Cloud Notebook (Google Colab GPU) | API Provider (Groq / OpenAI / AI Studio) |
| :--- | :--- | :--- | :--- |
| **Hardware Required** | User's local machine | Minimal local specs (runs in browser) | Minimal local specs (runs via HTTP) |
| **Privacy / Data Security** | **Highest** (100% on device) | Medium (stored temporarily on VM) | Lower (sent over network to provider) |
| **Cost** | Free (after purchasing hardware) | Free tier available / monthly compute units | Pay-per-token or free tier quotas |
| **Scalability** | Limited by laptop RAM/VRAM | Limited to single session GPU runtime | **Massive** (handled by enterprise clusters) |
| **Offline Support** | **Full offline support** | Requires stable internet | Requires stable internet |

---

### Questions & Answers

1. **If you had an 8 GB RAM laptop with no GPU, which approach would you pick?**
   - **Pick API Provider:** An 8 GB RAM laptop cannot run modern models comfortably without heavy quantization and severe lag. APIs offload 100% of the compute burden to server-side accelerators, delivering instant responses without heating up the laptop.

2. **Which approach is most private? Which is most scalable?**
   - **Most Private:** **Local Inference.** Data never leaves the local device, making it mandatory for strictly regulated clinical, financial, or confidential data.
   - **Most Scalable:** **API Provider.** Cloud providers automatically load-balance requests across thousands of enterprise GPUs with horizontal auto-scaling.

3. **When would you choose local over cloud, or the reverse?**
   - **Choose Local when:** Developing offline, complying with strict data privacy laws (GDPR/HIPAA), or building edge-device applications without recurring token costs.
   - **Choose Cloud/API when:** Running massive frontier models (70B+), needing high throughput/speed, or deploying production web apps without maintaining GPU server racks.