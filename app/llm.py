# app/llm.py
import platform
from llama_cpp import Llama

SYSTEM_PROMPT = (
    "You are Bearnard, the iACADEMY concierge AI. "
    "Answer clearly and briefly. If you don't know, just say so."
)

class LLM:
    def __init__(self):
        system = platform.system()

        if system == "Darwin":  # macOS (Metal)
            print("✅ macOS detected – using Metal GPU acceleration")
            self.model = Llama(
                model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
                n_ctx=2048,
                n_threads=6,
                n_gpu_layers=-1   # use all layers on GPU (Metal)
            )

        elif system == "Windows":  # Windows
            print("✅ Windows detected")
            # Try to use GPU if available (CUDA version of llama-cpp must be installed manually)
            self.model = Llama(
                model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
                n_ctx=2048,
                n_threads=8,
                n_gpu_layers=0   # set to -1 if using GPU CUDA build
            )

        else:  # Linux or others
            print("✅ Linux/Other detected – CPU mode")
            self.model = Llama(
                model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
                n_ctx=2048,
                n_threads=6,
                n_gpu_layers=0
            )

    def ask(self, prompt: str) -> str:
        response = self.model(
            prompt,
            max_tokens=150,
            temperature=0.6,
            stop=["User:", "Assistant:"]
        )
        return response["choices"][0]["text"].strip()
