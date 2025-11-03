import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class VoiceInput:
    def __init__(self, device=None):
        self.model = WhisperModel("tiny.en", device="cpu")
        self.sample_rate = 16000
        self.duration = 3  # seconds to record
        self.device = device  # microphone index

    def record_and_transcribe(self):
        print("🎙 Recording from microphone device:", self.device)

        audio = sd.rec(
            int(self.sample_rate * self.duration),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device
        )
        sd.wait()

        audio = np.squeeze(audio)
        segments, _ = self.model.transcribe(audio)
        text = " ".join(seg.text for seg in segments).strip()
        return text
