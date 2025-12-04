import sounddevice as sd
import numpy as np
import queue
import time

class VoiceInput:
    def __init__(self, model, device=None, sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        self.model = model 
        
        self.silence_threshold = 0.01   
        self.silence_duration = 1.2

    def adjust_for_ambient_noise(self, duration=1.0):
        print(f"\nCalibrating background noise on Device {self.device}...")
        try:
            rec = sd.rec(
                int(self.sample_rate * duration), 
                samplerate=self.sample_rate, 
                channels=1, 
                device=self.device, 
                dtype='float32'
            )
            sd.wait()
            rms = np.sqrt(np.mean(rec**2))
            self.silence_threshold = max(0.005, rms * 1.5)
            print(f"Threshold set to: {self.silence_threshold:.4f} (Noise Floor: {rms:.4f})")
        except Exception as e:
            print(f"Calibration failed: {e}. Using default threshold.")
            self.silence_threshold = 0.01

    def record_until_silence(self, callback=None, max_seconds=30):
        audio_queue = queue.Queue(maxsize=50)
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass 

        audio_buffer = []
        silence_timer = 0
        total_duration = 0
        chunk_duration = 0.1
        chunk_samples = int(self.sample_rate * chunk_duration)

        with sd.InputStream(samplerate=self.sample_rate, device=self.device, 
                            channels=1, dtype='float32', blocksize=chunk_samples,
                            callback=audio_callback):
            
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                chunk = chunk.flatten()
                audio_buffer.append(chunk)
                total_duration += chunk_duration

                vol = np.sqrt(np.mean(chunk**2))
                if callback:
                    callback(vol)
                
                is_talking = vol > self.silence_threshold
                if is_talking:
                    silence_timer = 0
                else:
                    silence_timer += chunk_duration

                if silence_timer >= self.silence_duration:
                    break
                
                if total_duration >= max_seconds:
                    print("\nMax recording duration reached.")
                    break

        return np.concatenate(audio_buffer, axis=0)

    def transcribe(self, audio_data):
        if audio_data is None or len(audio_data) == 0: 
            return ""
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        try:
            segments, _ = self.model.transcribe(
                audio_data, 
                beam_size=1,
                language="en",
                condition_on_previous_text=False, 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500), 
                initial_prompt="Hello, I am asking a question to the AI concierge."
            )
            
            text = " ".join([s.text for s in segments]).strip()
            return text
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""