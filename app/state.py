# app/state.py

from enum import Enum

class State(Enum):
    IDLE = 0              # Waiting for wake word "hey bearnard"
    WAKE_DETECTED = 1     # Wake word triggered
    LISTENING = 2         # Recording user's voice
    THINKING = 3          # Processing RAG + LLM
    SPEAKING = 4          # Responding (TTS)
