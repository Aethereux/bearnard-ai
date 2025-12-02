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
        self.wake_variants = [
            "hey bearnard", "hey bernard", "hey burner", 
            "ok bearnard", "okay bernard", "bearnard", "bernard", "hey"
        ]
        
        # Sliding Window: Keep last 2 seconds
        self.buffer_duration = 2.0  
        self.chunk_duration = 0.1   
        
        chunks_in_buffer = int(self.buffer_duration / self.chunk_duration)
        self.audio_buffer = collections.deque(maxlen=chunks_in_buffer)
        
        # LOWERED THRESHOLD: More sensitive to quiet voices
        self.energy_threshold = 0.002 

    def listen_for_wake_word(self, timeout=None, callback=None):
        """
        Listens for wake word with visual feedback callback.
        """
        self.audio_buffer.clear()
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        start_time = time.time()

        while True:
            # GUI Timeout Check
            if timeout and (time.time() - start_time > timeout):
                return False

            # 1. Record Chunk
            try:
                chunk = sd.rec(
                    chunk_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=self.device
                )
                sd.wait()
                chunk = chunk.flatten()
            except Exception:
                time.sleep(0.05)
                continue

            self.audio_buffer.append(chunk)

            # 2. Visual Feedback (Volume Bar)
            vol = np.sqrt(np.mean(chunk**2))
            if callback:
                callback(vol)

            # 3. Skip Silence (Optimization)
            if vol < self.energy_threshold:
                continue 

            # 4. Fast Transcription
            full_audio = np.concatenate(self.audio_buffer)
            
            try:
                # beam_size=1 is crucial for speed here
                segments, _ = self.model.transcribe(
                    full_audio, 
                    beam_size=1, 
                    language="en",
                    condition_on_previous_text=False
                )
                
                text = " ".join(seg.text for seg in segments).strip().lower()
                text = text.replace(",", "").replace(".", "").replace("!", "")

                # Check for keywords
                if any(variant in text for variant in self.wake_variants):
                    print(f"\n✨ WAKE WORD DETECTED: '{text}'")
                    return True

            except Exception:
                continue