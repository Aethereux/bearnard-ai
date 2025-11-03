# Bearnard – iACADEMY Concierge AI (Offline & Local)

Bearnard is an offline AI concierge that runs **entirely on your laptop**, with no APIs and no internet.
It can answer questions about the school, use your local data, listen to your voice, and even talk back.

---

## 1. Download the AI Model (Mistral 7B)

Download this file:

```
https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true
```

Then place it inside a folder named `models` in your project:

```
bearnard-ai/
 ├─ models/
 │   └─ mistral-7b-v0.1.Q4_K_M.gguf
 ├─ app/
 └─ README.md
```

---

## 2. Create Virtual Environment

### **macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install llama-cpp-python chromadb sentence-transformers faster-whisper pyttsx3 sounddevice torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**macOS only (if microphone errors):**

```bash
brew install portaudio
```

---

## 4. Run Bearnard

```bash
python app/main.py
```

You will be asked to choose:

* **1 = Voice Mode (wake word + mic)**
* **2 = Text Mode (type questions)**
  If Voice Mode is selected, you'll also pick your **microphone device**.

Then say:
**“Hey Bearnard”** to wake it up.

---

## What’s in This Stack?

| Component               | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| `llama-cpp-python`      | Runs Mistral-7B fully offline               |
| `chromadb`              | Local database for school information (RAG) |
| `sentence-transformers` | Embeds text for search                      |
| `faster-whisper`        | Converts speech → text offline              |
| `sounddevice`           | Records microphone input                    |
| `pyttsx3`               | Text-to-speech voice output                 |
| `torch`, `torchaudio`   | Audio + model backend                       |

---

## Future Plans

✔ Avatar + animated talking face (GUI) (PyQt6)
✔ Load directory/schedules into RAG
✔ Save conversation memory
✔ Touchscreen/info kiosk mode
