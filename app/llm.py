import platform
from llama_cpp import Llama

class LLM:
    def __init__(self):
        system = platform.system()
        
        ctx = 8192

        if system == "Darwin":  
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

    def ask(self, prompt: str, max_tokens: int = 256) -> str: 
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,  
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["[/INST]", "[INST]", "User:", "QUESTION:", "\n\n\n"]  
        )
        return response["choices"][0]["text"].strip()