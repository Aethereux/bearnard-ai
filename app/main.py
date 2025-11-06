import time
import sounddevice as sd
from state import State
from llm import LLM
from rag import Rag
from voice_input import VoiceInput
from voice_output import VoiceOutput
from wake_word import WakeWordDetector

def choose_mode():
    print("\nChoose Mode:")
    print("1 = 🎤 Voice Mode (Wake Word + Mic)")
    print("2 = ⌨️  Text Mode (Type your questions)")
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
    context = "\n\n".join(context_docs)
    return f"""
You are Bearnard, the concierge AI of iACADEMY.
Use the context below ONLY if it's relevant to answer the user's question.
If you don't know the answer, just say so briefly.

Context:
{context}

User: {user_query}
Assistant:"""

def main():
    mode = choose_mode()
    mic_index = choose_microphone() if mode == "voice" else None

    llm = LLM()
    rag = Rag(build_if_empty=True)
    ear = VoiceInput(device=mic_index)
    mouth = VoiceOutput()
    wake = WakeWordDetector(device=mic_index)

    state = State.IDLE
    user_text = ""
    answer = ""

    print("\nBearnard is running locally. Say 'Hey Bearnard' or type a question.\n")

    while True:
        if mode == "text":
            user_text = input("You: ")
            state = State.THINKING

        elif mode == "voice" and state == State.IDLE:
            if wake.listen_for_wake_word():
                print("✅ Wake word detected.")
                state = State.WAKE_DETECTED
            continue

        if state == State.WAKE_DETECTED:
            print("🎤 Listening for your question...")
            state = State.LISTENING

        if state == State.LISTENING:
            audio = ear.record_until_silence()
            user_text = ear.transcribe(audio)
            print(f"🗣 You said: {user_text}")
            state = State.THINKING

        if state == State.THINKING:
            print("🤔 Thinking...")
            docs = rag.search(user_text, n_results=3)
            prompt = build_prompt(user_text, docs)
            answer = llm.ask(prompt)
            state = State.SPEAKING

        if state == State.SPEAKING:
            print(f"\nBearnard: {answer}\n")
            if mode == "voice":
                mouth.speak(answer)
            state = State.IDLE

        time.sleep(0.1)

if __name__ == "__main__":
    main()
