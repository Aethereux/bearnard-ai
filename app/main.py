import time
import sounddevice as sd
from state import State
from llm import LLM
from rag import Rag
from voice_input import VoiceInput
from voice_output import VoiceOutput
from wake_word import WakeWordDetector

# -------------------------------
# 1. Mode Selection (Voice/Text)
# -------------------------------
def choose_mode():
    print("\nChoose Mode:")
    print("1 = 🎤 Voice Mode (Wake Word + Mic)")
    print("2 = ⌨️  Text Mode (Keyboard Only)")
    while True:
        choice = input("\nEnter 1 or 2: ")
        if choice == "1":
            print("✅ Voice Mode Selected")
            return "voice"
        elif choice == "2":
            print("✅ Text Mode Selected")
            return "text"
        else:
            print("❌ Invalid. Try again.")

# -------------------------------
# 2. Microphone Selection
# -------------------------------
def choose_microphone():
    print("\nAvailable Input Devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']} (Channels: {dev['max_input_channels']})")

    while True:
        try:
            choice = int(input("\nSelect microphone device index: "))
            if 0 <= choice < len(devices) and devices[choice]['max_input_channels'] > 0:
                print(f"✅ Selected Mic: {devices[choice]['name']}\n")
                return choice
            else:
                print("❌ Invalid choice. Try again.")
        except ValueError:
            print("❌ Please enter a number.")

# -------------------------------
# 3. Build LLM + RAG Prompt
# -------------------------------
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


# -------------------------------
# 4. MAIN PROGRAM
# -------------------------------
def main():
    mode = choose_mode()  # "voice" or "text"

    mic_index = None
    if mode == "voice":
        mic_index = choose_microphone()

    # Initialize core components
    llm = LLM()
    rag = Rag(build_if_empty=True)
    ear = VoiceInput(device=mic_index)
    mouth = VoiceOutput()
    wake = WakeWordDetector(device=mic_index)

    state = State.IDLE
    user_text = ""
    answer = ""

    print("\nBearnard is running locally.\n")

    while True:
        # =====================
        # TEXT MODE OPERATION
        # =====================
        if mode == "text":
            user_text = input("\nYou: ")
            
            if user_text.lower() in ["exit", "quit"]:
                print("👋 Goodbye.")
                break

            # Switch modes any time
            if user_text.lower() == ":voice":
                mode = "voice"
                mic_index = choose_microphone()
                wake = WakeWordDetector(device=mic_index)
                ear = VoiceInput(device=mic_index)
                state = State.IDLE
                print("🎤 Switched to Voice Mode")
                continue

            state = State.THINKING

        # =====================
        # VOICE MODE: IDLE
        # =====================
        if mode == "voice" and state == State.IDLE:
            if wake.listen_for_wake_word():
                print("✅ Wake word detected.")
                state = State.WAKE_DETECTED
            continue

        if state == State.WAKE_DETECTED:
            print("🎤 Listening for your question...")
            state = State.LISTENING

        # =====================
        # VOICE MODE: LISTENING
        # capture + transcribe
        # =====================
        if state == State.LISTENING:
            audio = ear.record_until_silence()
            user_text = ear.transcribe(audio)
            print(f"🗣 You said: {user_text}")

            if not user_text.strip():
                print("❌ No question detected. Returning to IDLE.")
                state = State.IDLE
                continue

            state = State.THINKING

        # =====================
        # THINKING (Common)
        # =====================
        if state == State.THINKING:
            print("🤔 Thinking...")
            docs = rag.search(user_text, n_results=3)
            prompt = build_prompt(user_text, docs)
            answer = llm.ask(prompt)
            state = State.SPEAKING

        # =====================
        # SPEAK ANSWER
        # =====================
        if state == State.SPEAKING:
            print(f"\nBearnard: {answer}\n")
            if mode == "voice":
                mouth.speak(answer)
            state = State.IDLE

        time.sleep(0.1)

if __name__ == "__main__":
    main()
