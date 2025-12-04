import numpy as np
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

class MLXSegment:
    def __init__(self, text):
        self.text = text

class MLXWhisperWrapper:
    def __init__(self, model_path="mlx-community/whisper-large-v3-turbo"):
        if mlx_whisper is None:
            raise ImportError("mlx_whisper is not installed.")
            
        self.model_path = model_path
        print(f"MLX Model Ready: {model_path}")
        
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(dummy_audio, path_or_hf_repo=self.model_path)
        except Exception:
            pass

    def transcribe(self, audio, beam_size=1, language="en", **kwargs):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=language,
            # MLX uses different params than Faster-Whisper, we map the critical ones
            verbose=False
        )
        
        # Convert dict result to the object format Bearnard expects
        class Segment:
            def __init__(self, text):
                self.text = text
        
        text = result.get("text", "").strip()
        return [MLXSegment(text)], None