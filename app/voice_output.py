# app/voice_output.py

import pyttsx3

class VoiceOutput:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)  # adjust speaking speed if needed

    def speak(self, text: str):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()
