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
        
        # Reduced variants list
        self.wake_variants = [
            "hey bearnard", "hey bernard", "ok bearnard", "okay bernard", "bearnard"
        ]
        
        # BUFFER SETTINGS
        self.buffer_duration = 2.0  
        self.chunk_duration = 0.2   
        
        # INFERENCE INTERVAL: Run inference every 3 chunks (approx 0.6s)
        self.inference_interval = 3 
        self.chunk_counter = 0
        
        chunks_in_buffer = int(self.buffer_duration / self.chunk_duration)
        self.audio_buffer = collections.deque(maxlen=chunks_in_buffer)
        
        self.energy_threshold = 0.002 
        
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Status: {status}")
        self.audio_queue.put(indata.copy())

    def start_stream(self):
        if self.is_listening: return
        self.audio_queue = queue.Queue()
        self.audio_buffer.clear()
        self.chunk_counter = 0
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
            print(f"Error starting stream: {e}")

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
            # 1. Timeout Check
            if timeout and (time.time() - start_time > timeout):
                return False

            try:
                # 2. NON-BLOCKING GET
                chunk = self.audio_queue.get_nowait()
                chunk = chunk.flatten()
                
                self.audio_buffer.append(chunk)
                
                vol = np.sqrt(np.mean(chunk**2))
                if volume_callback:
                    volume_callback(vol)

                # 3. LAG PROTECTION
                # If queue is backing up, skip processing to catch up
                if self.audio_queue.qsize() > 2:
                    continue

                # 4. ENERGY CHECK
                if vol < self.energy_threshold:
                    continue 

                # 5. INTERVAL CHECK
                self.chunk_counter += 1
                if self.chunk_counter % self.inference_interval != 0:
                    continue

                # 6. RUN INFERENCE
                full_audio = np.concatenate(self.audio_buffer)
                
                if len(full_audio) < 16000:
                    continue

                segments, _ = self.model.transcribe(
                    full_audio, 
                    beam_size=1, 
                    language="en", 
                    condition_on_previous_text=False,
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text = " ".join(s.text for s in segments).strip().lower()
                clean_text = text.replace(",", "").replace(".", "").replace("!", "").strip()

                # --- NEW FILTER APPLIED HERE ---
                # Reject if silence, common hallucinations, OR if text is too long
                if not clean_text or clean_text in ["you", "thank you", "watching"]:
                    continue

                # STRICT LENGTH CHECK: Only accept short phrases < 15 chars
                if len(clean_text) >= 15:
                    continue
                # -------------------------------

                if transcript_callback:
                    transcript_callback(clean_text)

                if any(variant in clean_text for variant in self.wake_variants):
                    print(f"\nWAKE WORD DETECTED: '{clean_text}'")
                    self.stop_stream()
                    return True

            except queue.Empty:
                time.sleep(0.02)
                continue
            except Exception as e:
                print(f"Wake Word Error: {e}")
                continue