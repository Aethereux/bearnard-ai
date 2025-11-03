import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class WakeWordDetector:
    def __init__(self, wake_phrase="hey bearnard", device=None):
        self.wake_phrase = wake_phrase.lower()
        self.sample_rate = 16000
        self.duration = 1.5  # seconds
        self.device = device  # microphone index
        self.model = WhisperModel("tiny.en", device="cpu")

    def listen_for_wake_word(self):
        print(f"🎧 Listening for '{self.wake_phrase}' using mic device: {self.device}")

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
        text = " ".join(seg.text for seg in segments).strip().lower()

        if text:
            print(f"🗣 Heard: {text}")

        wake_variants = ["hey bearnard", "hey bernard", "hey bear nard", "hey bernard"]
        return any(word in text for word in wake_variants)
