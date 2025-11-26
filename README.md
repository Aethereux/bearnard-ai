# 🎓 Bearnard – iACADEMY Concierge AI (Offline & Local)

**Welcome to Bearnard\!** This is a fully offline, privacy-first AI concierge designed for **iACADEMY (The Nexus)**.

<<<<<<< HEAD
---
## For Systems w/o C++ Compiler
Install the vs_Build tools through: https://visualstudio.microsoft.com/downloads/
=======
Unlike Siri or ChatGPT, Bearnard runs **100% on your laptop hardware**. He has no API keys, no monthly fees, and he doesn't send your voice to the cloud. He uses a **Two-Stage Wake Word Pipeline** to act as a strict librarian for school questions (using RAG) but switches to a chill "Game Changer" ambassador for general interactions.
>>>>>>> 6e8430c36d8f4f89948e2b0be0cf549760e4c122

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
    │   ├── wake_word.py
    │   └── ...
    └── README.md
    ```

-----

## 2\. Installation

Set up your environment. Bearnard relies on optimized audio processing and vector databases.

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

Create a file named `requirements.txt` in your folder with this content:

```text
llama-cpp-python
chromadb
sentence-transformers
faster-whisper
pyttsx3
sounddevice
numpy
pypdf
```
note: -- index-url no longer needs to be installed (atm)

Then run:

```bash
pip install -r requirements.txt
```

-----

### ⚠️ Troubleshooting (Read if Install Fails)

**1. Mac M1/M2/M3 Users (`llama-cpp-python` error)**
If installation fails, your Mac likely needs to compile the AI engine specifically for Apple Metal (GPU). Run this specific command:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

**2. Microphone Errors (`PortAudio`)**
If you see errors about "PortAudio" or "No module named sounddevice" on Mac:

```bash
brew install portaudio
```

*(If you don't have Homebrew, install it from brew.sh first)*

-----

## 3\. How to Run Bearnard

Once everything is installed, wake him up:

```bash
python app/main.py
```

### **The Modes**

When you start the app, you can choose how to interact:

  * **1 = 🎤 Voice Mode:** The full "Smart Speaker" experience.
      * **The Sentry:** He sleeps until you say **"Hey Bearnard"**.
      * **The Feedback:** You will hear a **BEEP** (`\a`) when he is ready.
      * **The Recorder:** Watch the **Live Volume Bar** (`Vol: 0.05 |||||`) to verify he hears you.
  * **2 = ⌨️ Text Mode:** Perfect for testing RAG data. You type questions, and he replies via text **and** voice.

-----

## 🧠 Under the Hood: The Architecture

We scrapped the complex Neural Network VADs for a more robust, "Bare Metal" approach used by industry smart speakers.

| Component | Architecture | Why it's better |
| :--- | :--- | :--- |
| **The Sentry** (`wake_word.py`) | **Sliding Window Ring Buffer** | It keeps the last 2 seconds of audio in memory. Even if you pause mid-sentence ("Hey... Bearnard"), it catches it. Includes an **Energy Gate** to save CPU when silent. |
| **The Ears** (`voice_input.py`) | **Visual Energy Gate** | Uses mathematical volume calculation (RMS) instead of AI. Features a **Live Visual Bar** so you can see exactly what the mic hears. |
| **The Brain** (`llm.py`) | **Mistral 7B (Quantized)** | Runs locally. Uses a dynamic token limit (switches between short answers and long lists based on context). |
| **The Memory** (`rag.py`) | **ChromaDB + Prose** | Scans documents for semantic meaning. We optimized the data to use **Natural Language** (sentences) instead of lists for better retrieval. |
| **The Core** (`main.py`) | **Shared Model Instance** | We load the heavy Whisper AI **once** and pass it to both the Wake Word detector and the Recorder to save RAM and reduce latency. |

-----

## 📚 Adding Your Own Data (RAG)

Want to teach Bearnard about your specific schedule, thesis guidelines, or canteen menu?

1.  Put your `.pdf` or `.txt` files inside the `data/` folder.
2.  **Format Tip:** Do NOT use raw lists (e.g., `Location: Room 1, Room 2`).
      * **Bad:** `Office: OSAS`
      * **Good:** `The OSAS (Office of Student Affairs) is located on the Mezzanine level.`
3.  **Important:** If you add new files, **delete the `chroma_db` folder**. Bearnard will automatically rebuild his brain index the next time you run him.

-----

## 🔮 Roadmap

  * [x] **Smart RAG:** Now supports PDFs and Semantic Chunking.
  * [x] **Robust Wake Word:** Zero-loss Sliding Window Buffer.
  * [x] **Visual Debugging:** Live volume metering in the console.
  * [ ] **GUI:** A PyQt6 interface with an animated bear avatar.
  * [ ] **Vision:** Ability to "see" via webcam (using LLaVA models).

-----

*Built with ❤️ by the Nexus Team.*