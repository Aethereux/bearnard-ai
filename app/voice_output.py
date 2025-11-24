import pyttsx3

class VoiceOutput:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Adjust speed (Standard is usually ~200, 175 is clearer)
        self.engine.setProperty('rate', 175)  

    def speak(self, text: str):
        if not text:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")