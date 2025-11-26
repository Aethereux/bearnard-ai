from enum import Enum

class State(Enum):
    IDLE = 0              
    WAKE_DETECTED = 1     
    LISTENING = 2         
    THINKING = 3          
    SPEAKING = 4