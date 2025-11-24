# 🎓 Bearnard – iACADEMY Concierge AI (Offline & Local)

**Welcome to Bearnard\!** This is a fully offline, privacy-first AI concierge designed for **iACADEMY (The Nexus)**.

Unlike Siri or ChatGPT, Bearnard runs **100% on your laptop hardware**. He has no API keys, no monthly fees, and he doesn't send your voice to the cloud. He uses a "Decision Protocol" to act as a strict librarian for school questions (using RAG) but switches to a chill "Game Changer" ambassador for general interactions.

-----

## 1\. Get the Brain (The Model)

Bearnard uses **Mistral 7B Instruct**, a powerful open-source model. You need to download it manually because it's too big for GitHub (\~4GB).

1.  **Download this specific file (Quantized for speed):**
    👉 [Download mistral-7b-instruct-v0.1.Q4\_K\_M.gguf](https://www.google.com/search?q=https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf%3Fdownload%3Dtrue)

2.  **Place it in your project:**
    Create a `models/` folder and drop the file there. Your structure should look like this:

    ```text
    bearnard-ai/
    ├── data/             <-- Put your School PDFs/Text files here
    ├── models/
    │   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf  <-- The file you downloaded
    ├── app/
    │   ├── main.py
    │   └── ...
    └── README.md
    ```

-----

## 2\. Installation

Set up your environment. Bearnard relies on some heavy hitters like PyTorch and ChromaDB.

### **Step A: Virtual Environment**

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### **Step B: Install Dependencies**

**Option 1: The Easy Way (Recommended)**
Create a file named `requirements.txt` in your folder with this content:

```text
llama-cpp-python
chromadb
sentence-transformers
faster-whisper
pyttsx3
sounddevice
pypdf
torch
torchaudio
```

Then run:

```bash
pip install -r requirements.txt
```

**Option 2: The Manual Way**

```bash
pip install llama-cpp-python chromadb sentence-transformers faster-whisper pyttsx3 sounddevice pypdf torch torchaudio
```

-----

### ⚠️ Troubleshooting (Read if Install Fails)

**1. Mac M1/M2/M3 Users (`llama-cpp-python` error)**
If installation fails, your Mac likely needs to compile the AI engine specifically for Apple Metal (GPU). Run this specific command:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

*After running this, try `pip install -r requirements.txt` again.*

**2. Microphone Errors (`PortAudio`)**
If you see errors about "PortAudio" or "No module named sounddevice" on Mac:

```bash
brew install portaudio
```

-----

## 3\. How to Run Bearnard

Once everything is installed, wake him up:

```bash
python app/main.py
```

### **The Modes**

When you start the app, you can choose how to interact:

  * **1 = 🎤 Voice Mode:** The full sci-fi experience. He listens for **"Hey Bearnard"**, records your question, and speaks the answer back.
  * **2 = ⌨️ Text Mode:** Perfect for testing. You type questions, and he replies via text **and** voice.

-----

## 🧠 Under the Hood: How it Works

We built a custom architecture to make Bearnard smart but fast.

| Component | What it does |
| :--- | :--- |
| **The Brain** (`llama-cpp`) | Runs **Mistral 7B Instruct** on your CPU/GPU. It uses a "Decision Protocol" to decide if it should answer directly or look up data. |
| **The Ears** (`voice_input`) | Uses **Silero VAD** (Voice Activity Detection) to detect when you stop speaking instantly, and **Whisper** to transcribe audio. |
| **The Memory** (`rag.py`) | Uses **ChromaDB** and **pypdf**. It reads your PDFs/Txt files, chops them into smart paragraphs, and finds the exact answer to user questions. |
| **The Voice** (`pyttsx3`) | A lightweight text-to-speech engine that works offline. |

-----

## 📚 Adding Your Own Data (RAG)

Want to teach Bearnard about your specific schedule, thesis guidelines, or canteen menu?

1.  Put your `.pdf` or `.txt` files inside the `data/` folder.
2.  **Format Tip:** For text files, use headers like `TOPIC: HISTORY` or `LOCATION: 5TH FLOOR` to help him understand context.
3.  **Important:** If you add new files, **delete the `chroma_db` folder**. Bearnard will automatically rebuild his brain index the next time you run him.

-----

## 🔮 Roadmap

  * [x] **Smart RAG:** Now supports PDFs and Paragraph Chunking.
  * [x] **Silero VAD:** No more waiting 2 seconds for him to realize you stopped talking.
  * [ ] **GUI:** A PyQt6 interface with an animated bear avatar.
  * [ ] **Vision:** Ability to "see" via webcam (using LLaVA models).

-----

*Built with ❤️ by the Nexus Team.*