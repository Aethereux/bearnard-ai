import sounddevice as sd
import numpy as np
import time

class VoiceInput:
    def __init__(self, model, device=None, sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        self.model = model 
        
        # --- SETTINGS ---
        # Trigger volume (Based on your 0.05 voice level)
        self.silence_threshold = 0.01   
        # How long to wait after you stop talking before finishing
        self.silence_duration = 1.5     

    def record_until_silence(self):
        print("\n") # Spacing for visual bar
        
        audio_buffer = []
        silence_timer = 0
        
        # Fast updates (0.2s) for smooth visual animation
        chunk_duration = 0.2 
        chunk_samples = int(self.sample_rate * chunk_duration)

        while True:
            chunk = sd.rec(
                chunk_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device
            )
            sd.wait()
            chunk = chunk.flatten()
            audio_buffer.append(chunk)

            # Calculate Volume
            vol = np.abs(chunk).mean()
            is_talking = vol > self.silence_threshold
            
            # --- LIVE VISUAL DEBUG BAR ---
            # Scale volume for display
            bar_len = int(vol * 500) 
            bar = "█" * min(bar_len, 50) 
            
            if is_talking:
                status = "🔴 REC"
                silence_timer = 0
            else:
                status = "⏳ ..."
                silence_timer += chunk_duration
            
            # \r allows us to overwrite the line
            print(f"\r🎤 {status} | Vol: {vol:.4f} | {bar:<50}", end="", flush=True)
            # -----------------------------

            if silence_timer >= self.silence_duration:
                print("\n🛑 Silence detected. Processing...")
                break

        return np.concatenate(audio_buffer, axis=0)

    def transcribe(self, audio_data):
        if audio_data is None: return ""
        # Higher beam_size = better accuracy for the actual question
        segments, _ = self.model.transcribe(audio_data, beam_size=5)
        return " ".join([s.text for s in segments]).strip()