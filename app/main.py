import time
import datetime
import sounddevice as sd
from faster_whisper import WhisperModel
from state import State
from llm import LLM
from rag import Rag
from voice_input import VoiceInput
from voice_output import VoiceOutput
from wake_word import WakeWordDetector

# --- REMOVED WAKEWORD IMPORT ---

def choose_mode():
    print("\nChoose Mode:")
    print("1 = 🎤 Voice Mode (Continuous Conversation)")
    print("2 = ⌨️  Text Mode (Type + Voice Output)")
    while True:
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            return "voice"
        elif choice == "2":
            return "text"
        else:
            print("❌ Invalid. Try again.")

def choose_microphone():
    print("\n🎤 Available Input Devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']} (Channels: {dev['max_input_channels']})")

    while True:
        try:
            choice = int(input("\nSelect mic device index: "))
            if 0 <= choice < len(devices) and devices[choice]['max_input_channels'] > 0:
                print(f"✅ Selected: {devices[choice]['name']}")
                return choice
        except ValueError:
            pass
        print("❌ Invalid. Try again.")


def build_prompt(user_query: str, context_docs: list[str]) -> str:
    if context_docs:
        formatted_context = "\n---\n".join(context_docs)
    else:
        formatted_context = "NO_DATA_FOUND"

    current_time = datetime.datetime.now().strftime("%A, %I:%M %p")

    return f"""[INST] You are Bearnard, the AI Concierge of iACADEMY (The Nexus).
Current Time: {current_time}

### INSTRUCTIONS:
1. **SOURCE OF TRUTH:** specific answer is found in the [CONTEXT] block below, use it.
2. **UNKNOWN INFO:** If the [CONTEXT] contains "NO_DATA_FOUND" or does not contain the answer, say exactly: "I'm sorry, I don't have that information in my current records."
3. **OFF-TOPIC:** If the user asks about math, coding, or general world trivia (not related to iACADEMY), politely decline.
4. **VOICE OPTIMIZATION:** You are speaking to the user.
   - Keep answers **short** (under 2 sentences if possible).
   - Do NOT use lists, bullet points, or markdown formatting.
   - If listing items, separate them with commas for natural speech.

### [CONTEXT]
{formatted_context}

### [USER QUESTION]
{user_query}

### [BEARNARD'S ANSWER]
[/INST]"""

def main():
    mode = choose_mode()
    mic_index = choose_microphone() if mode == "voice" else None

    print("\n⏳ Loading Whisper (Shared)...")
    # 'base.en' is the sweet spot for accuracy/speed on local CPU
    shared_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("✅ Whisper Loaded.")

    llm = LLM()
    rag = Rag(build_if_empty=True)
    
    # Initialize both Ear (Recorder) and Wake (Sentry)
    ear = VoiceInput(model=shared_whisper, device=mic_index)
    wake = WakeWordDetector(model=shared_whisper, device=mic_index)
    mouth = VoiceOutput()
    
    state = State.IDLE

    print("\n🐻 Bearnard is ready. Say 'Hey Bearnard'.\n")

    while True:
        # --- PHASE 1: WAKE WORD ---
        if mode == "voice" and state == State.IDLE:
            # Blocks here until wake word is heard
            if wake.listen_for_wake_word():
                print("\a") # BEEP SOUND
                state = State.LISTENING
            continue

        # --- PHASE 2: RECORD QUESTION ---
        if mode == "voice" and state == State.LISTENING:
            audio = ear.record_until_silence()
            
            print("📝 Transcribing...")
            user_text = ear.transcribe(audio)
            print(f"🗣 You said: {user_text}")
            
            if not user_text.strip():
                print("🤷 Heard nothing.")
                state = State.IDLE
                continue
            state = State.THINKING
            
        elif mode == "text":
            user_text = input("You: ")
            state = State.THINKING

        # --- PHASE 3: THINK & SPEAK ---
        if state == State.THINKING:
            print("🤔 Thinking...")
            docs = rag.search(user_text)
            
            # Dynamic Token Limit (Short answers normally, Long for lists)
            token_limit = 1024 if "list" in user_text.lower() else 256
            
            prompt = build_prompt(user_text, docs)
            answer = llm.ask(prompt, max_tokens=token_limit)
            
            print(f"\n🐻 Bearnard: {answer}\n")
            mouth.speak(answer)
            
            state = State.IDLE # Go back to sleep

        time.sleep(0.1)

if __name__ == "__main__":
    main()