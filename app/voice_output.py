import pyttsx3
import platform
import subprocess

class VoiceOutput:
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
        
        if not self.is_mac:
            # Initialize pyttsx3 only on non-Mac systems to avoid threading issues
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 175)  

    def speak(self, text: str):
        if not text:
            return
        
        if self.is_mac:
            # Use native macOS 'say' command which is thread-safe and reliable
            try:
                subprocess.run(['say', '-r', '175', text])
            except Exception as e:
                print(f"Mac TTS Error: {e}")
        else:
            # Standard pyttsx3 for Windows/Linux
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")