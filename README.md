```markdown
# 🎓 Bearnard – iACADEMY Concierge AI (Offline & Local)

Bearnard is an offline AI concierge that runs **entirely on your laptop**, with no APIs and no internet.
It can answer questions about the school, search local data (RAG), listen to your voice, and even talk back.

---

## ✅ 1. Download the AI Model (Mistral 7B)

Download this file:

```

[https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true)

```

Create a `models/` folder in the project and place the file inside:

```

bearnard-ai/
├─ models/
│   └─ mistral-7b-v0.1.Q4_K_M.gguf
├─ app/
└─ README.md

````

---

## ✅ 2. Create Virtual Environment

### **macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
````

### **Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

## ✅ 3. Install Dependencies

```bash
pip install llama-cpp-python chromadb sentence-transformers faster-whisper pyttsx3 sounddevice torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**macOS only (if microphone errors):**

```bash
brew install portaudio
```

---

## ✅ 4. Run Bearnard

```bash
python app/main.py
```

When it starts, you will be asked:

✔ **Choose mode:**

* `1 = Voice Mode` (Wake word + Microphone)
* `2 = Text Mode` (Type questions manually)

✔ If Voice Mode is chosen → it will list all microphone devices → you choose one.

Then say:
**“Hey Bearnard”** to wake it up.

---

## ✅ AI Logic & System Components

| Component               | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `llama-cpp-python`      | Runs Mistral-7B locally (no internet)      |
| `chromadb`              | Vector database for RAG (school knowledge) |
| `sentence-transformers` | Embeds text data for search                |
| `faster-whisper`        | Speech-to-text (offline voice input)       |
| `sounddevice`           | Microphone recording                       |
| `pyttsx3`               | Text-to-speech voice output                |
| `torch`, `torchaudio`   | Required backend for whisper/audio         |

---

## ✅ Future Enhancements (Planned)

✔ Avatar + animated talking face (PyQt6 GUI)
✔ Load real school directory / schedules into RAG
✔ Save user questions & memory
✔ Touchscreen / Kiosk mode support
✔ Auto-detect macOS vs Windows for optimized LLM performance (Metal / CUDA / CPU threads)

---

💡 *Everything runs locally — no API keys, no internet connection required.*

```
