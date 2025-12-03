import mlx_whisper

class MLXWhisperWrapper:
    """
    Adapts mlx-whisper to look like faster-whisper for Bearnard.
    """
    def __init__(self, model_path="mlx-community/whisper-large-v3-turbo"):
        self.model_path = model_path
        print(f"MLX Model Ready: {model_path}")

    def transcribe(self, audio, beam_size=1, language="en", condition_on_previous_text=False, **kwargs):
        """
        Wraps the functional mlx_whisper.transcribe API to behave like an object method.
        """

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=language,
            verbose=False
        )
        
        class Segment:
            def __init__(self, text):
                self.text = text
        
        text = result.get("text", "").strip()
        return [Segment(text)], None