import sounddevice as sd
import numpy as np
import collections
import queue
import time

class WakeWordDetector:
    def __init__(self, model, device=None):
        self.sample_rate = 16000
        self.device = device
        self.model = model
        
        self.wake_variants = [
            "hey bearnard", "hey bernard", "hey burner", 
            "ok bearnard", "okay bernard", "bearnard", "bernard"
        ]
        
        # SLIDING WINDOW: 2.0 seconds
        self.buffer_duration = 2.0  
        self.chunk_duration = 0.2   
        
        chunks_in_buffer = int(self.buffer_duration / self.chunk_duration)
        self.audio_buffer = collections.deque(maxlen=chunks_in_buffer)
        
        self.energy_threshold = 0.002 
        
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start_stream(self):
        if self.is_listening: return
        self.audio_queue = queue.Queue()
        self.audio_buffer.clear()
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device,
                channels=1,
                dtype='float32',
                blocksize=int(self.sample_rate * self.chunk_duration),
                callback=self._callback
            )
            self.stream.start()
            self.is_listening = True
            print("Wake Word Stream Started...")
        except Exception as e:
            print(f"Error: {e}")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_listening = False

    def listen_for_wake_word(self, timeout=None, volume_callback=None, transcript_callback=None):
        if not self.is_listening:
            self.start_stream()

        start_time = time.time()

        while True:
            if timeout and (time.time() - start_time > timeout):
                return False

            try:
                chunk = self.audio_queue.get_nowait()
                chunk = chunk.flatten()
                
                self.audio_buffer.append(chunk)
                
                vol = np.sqrt(np.mean(chunk**2))
                if volume_callback:
                    volume_callback(vol)

                if vol < self.energy_threshold:
                    continue 

                full_audio = np.concatenate(self.audio_buffer)
                if len(full_audio) < 16000:
                    continue

                segments, _ = self.model.transcribe(
                    full_audio, 
                    beam_size=1, 
                    language="en", 
                    condition_on_previous_text=False
                )
                
                text = " ".join(s.text for s in segments).strip().lower()
                clean_text = text.replace(",", "").replace(".", "").replace("!", "").strip()

                if len(clean_text) > 15:
                    continue

                if not clean_text or clean_text in ["you", "thank you", "watching"]:
                    continue

                if transcript_callback:
                    transcript_callback(clean_text)

                if any(variant in clean_text for variant in self.wake_variants):
                    print(f"\nWAKE WORD DETECTED: '{clean_text}'")
                    self.stop_stream()
                    return True

            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception:
                continue