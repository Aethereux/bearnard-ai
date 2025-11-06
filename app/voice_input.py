import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class VoiceInput:
    def __init__(self, device=None, sample_rate=16000, silence_threshold=0.01, silence_duration=1.0):
        self.device = device
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold  # Volume threshold to detect silence
        self.silence_duration = silence_duration    # Seconds of silence before stopping
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    def is_silent(self, audio_chunk):
        return np.abs(audio_chunk).mean() < self.silence_threshold

    def record_until_silence(self):
        print("🎧 Recording... (stop speaking to finish)")

        audio_buffer = []
        silence_time = 0

        while True:
            chunk = sd.rec(
                int(self.sample_rate * 0.3),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device
            )
            sd.wait()
            chunk = chunk.flatten()
            audio_buffer.append(chunk)

            # Check for silence
            if self.is_silent(chunk):
                silence_time += 0.3
                if silence_time >= self.silence_duration:
                    print("🛑 Silence detected. Processing...")
                    break
            else:
                silence_time = 0

        audio_data = np.concatenate(audio_buffer, axis=0)
        return audio_data

    def transcribe(self, audio_data):
        segments, _ = self.model.transcribe(audio_data)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
