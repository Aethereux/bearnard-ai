import sounddevice as sd
import numpy as np
import collections
import time

class WakeWordDetector:
    def __init__(self, model, device=None):
        self.sample_rate = 16000
        self.device = device
        self.model = model
        
        # --- TUNING ---
        # "Bearnard" is tricky for AI. We add common misinterpretations.
        self.wake_variants = [
            "hey bearnard", "hey bernard", "hey burner", 
            "ok bearnard", "okay bernard", "bearnard", "bernard", "hey"
        ]
        
        # 1. SLIDING WINDOW SETTINGS
        self.buffer_duration = 2.0  # We always listen to the LAST 2 seconds
        self.chunk_duration = 0.25  # We update the buffer 4 times a second (Fast response)
        
        # Calculate buffer size
        chunks_in_buffer = int(self.buffer_duration / self.chunk_duration)
        self.audio_buffer = collections.deque(maxlen=chunks_in_buffer)
        
        # 2. ENERGY GATE (CPU Saver)
        # Threshold below which we don't even bother checking for words
        self.energy_threshold = 0.005 

    def listen_for_wake_word(self, timeout=None):
        # print("\n💤 Waiting for wake word...", end="", flush=True) # Optional: comment out to reduce spam
        self.audio_buffer.clear()
        
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        start_time = time.time()

        while True:
            # --- NEW: Timeout Check ---
            if timeout and (time.time() - start_time > timeout):
                return False
            # --------------------------

            # 1. Record small chunk
            chunk = sd.rec(
                chunk_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device
            )
            sd.wait()
            chunk = chunk.flatten()

            # 2. Add to Ring Buffer
            self.audio_buffer.append(chunk)

            # 3. Energy Gate Check
            vol = np.sqrt(np.mean(chunk**2))
            if vol < self.energy_threshold:
                continue 

            # 4. Transcribe the Ring Buffer
            full_audio = np.concatenate(self.audio_buffer)
            
            try:
                segments, _ = self.model.transcribe(
                    full_audio, 
                    beam_size=1, 
                    language="en",
                    condition_on_previous_text=False
                )
                
                text = " ".join(seg.text for seg in segments).strip().lower()
                text = text.replace(",", "").replace(".", "").replace("!", "")

                if any(variant in text for variant in self.wake_variants):
                    print(f"\n✨ WAKE WORD DETECTED: '{text}'")
                    return True

            except Exception:
                continue