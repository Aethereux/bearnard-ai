# app/llm.py
from llama_cpp import Llama

class LLM:
    def __init__(self):
        self.model = Llama(
        model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1    # ✅ -1 = use Metal GPU for all layers
    )


    def ask(self, prompt: str) -> str:
        out = self.model(
            prompt,
            max_tokens=300,
            temperature=0.6,
            stop=["User:", "Assistant:", "###", "\nContext"]
        )
        return out["choices"][0]["text"].strip()

