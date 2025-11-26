import platform
from llama_cpp import Llama

class LLM:
    def __init__(self):
        system = platform.system()
        
        # Set Context Window to 8192 to hold all school data
        ctx = 8192

        if system == "Darwin":  # macOS
            print("macOS detected – using Metal GPU acceleration")
            self.model = Llama(
                model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                n_ctx=ctx,
                n_threads=6,
                n_gpu_layers=-1
            )
        elif system == "Windows":
            print("Windows detected")
            self.model = Llama(
                model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                n_ctx=ctx,
                n_threads=8,
                n_gpu_layers=0 
            )
        else:
            print("Linux/Other detected")
            self.model = Llama(
                model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                n_ctx=ctx,
                n_threads=6,
                n_gpu_layers=0
            )

    # Change the ask method signature to accept max_tokens
    def ask(self, prompt: str, max_tokens: int = 256) -> str: 
        response = self.model(
            prompt,
            max_tokens=max_tokens,  # <--- Use the variable here
            temperature=0.2,
            stop=["[/INST]", "User:", "###"]
        )
        return response["choices"][0]["text"].strip()