import sys
import time
import datetime

from faster_whisper import WhisperModel
from rag import Rag
from llm import LLM
from voice_input import VoiceInput
from voice_output import VoiceOutput
from wake_word import WakeWordDetector

import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QScrollArea, QFrame, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QColor, QIcon, QFont

# (Background is now handled by the Stylesheet gradient)
WHITE_PANEL = "#ffffff"     # The Chat Card
INPUT_BG = "#1b1b3a"        # The Dark Blue Input Pill
BUTTON_BG = "#1f1f1f"       # The Dark Mode Buttons
TEXT_COLOR = "#333333"      # Chat Text Color

STYLESHEET = f"""
    QMainWindow {{
        /* Circular Gradient starting at Top-Left (0,0) */
        background: qradialgradient(
            spread:pad, cx:0, cy:0, radius:1.8, fx:0, fy:0,
            stop:0 #000b21, 
            stop:0.5 #081378, 
            stop:1 #1423a8
        );
    }}
    QWidget {{
        font-family: 'Segoe UI', sans-serif;
    }}
    
    /* RIGHT SIDE: THE WHITE CHAT CARD */
    #ChatCard {{
        background-color: {WHITE_PANEL};
        border-radius: 20px;
    }}
    
    /* MODE BUTTONS (Let's Talk / Chat) */
    QPushButton.ModeBtn {{
        background-color: {BUTTON_BG};
        color: white;
        border-radius: 15px;
        padding: 5px 20px;
        font-weight: bold;
        font-size: 15px;
        text-align: left;
    }}
    
    QPushButton.ModeBtn:hover {{
        background-color: #333333;
    }}

    /* --- CHAT HISTORY AREA --- */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QWidget#ChatContent {{
        background-color: transparent;
    }}
    QLabel.ChatMessage {{
        color: {TEXT_COLOR};
        font-size: 14px;
        padding: 5px;
    }}

    /* INPUT PILL (The dark bar at the bottom) */
    QFrame#InputPill {{
        background-color: {INPUT_BG};
        border-radius: 25px; /* Rounded Pill Shape */
    }}
    QLineEdit {{
        background-color: transparent;
        color: white;
        font-size: 14px;
        border: none;
    }}
    QLineEdit::placeholder {{
        color: #aaaaaa;
    }}
"""

# AI WORKER
class AIWorker(QThread):
    response_ready = pyqtSignal(str)
    state_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.mode = "chat"
        self.is_running = True
        self.input_queue = None

    def set_mode(self, mode):
        self.mode = mode

    def process_text(self, text):
        self.input_queue = text

    def run(self):
        self.state_changed.emit("LOADING")
        print("⏳ Loading Models...")
        
        shared_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
        self.rag = Rag(build_if_empty=True)
        self.llm = LLM()
        self.ear = VoiceInput(model=shared_whisper)
        self.wake = WakeWordDetector(model=shared_whisper)
        self.mouth = VoiceOutput()

        self.state_changed.emit("IDLE")
        print("✅ AI Ready.")

        while self.is_running:
            if self.input_queue:
                text = self.input_queue
                self.input_queue = None
                self.generate_response(text)
                continue

            if self.mode == "voice":
                # Ensure wake_word.py has 'timeout' param!
                if self.wake.listen_for_wake_word(timeout=0.5):
                    self.state_changed.emit("LISTENING")
                    audio = self.ear.record_until_silence()
                    self.state_changed.emit("THINKING")
                    user_text = self.ear.transcribe(audio)
                    if user_text.strip():
                        self.generate_response(user_text)
                    else:
                        self.state_changed.emit("IDLE")
            time.sleep(0.1)

    def generate_response(self, user_text):
        self.state_changed.emit("THINKING")
        docs = self.rag.search(user_text)
        
        # Log context results
        if docs:
            print(f"\n📚 [CONTEXT] Found {len(docs)} relevant documents:")
            for i, doc in enumerate(docs, 1):
                print(f"  [{i}] {doc[:100]}..." if len(doc) > 100 else f"  [{i}] {doc}")
            print()
            formatted_context = "\n---\n".join(docs)
        else:
            print("⚠️  [CONTEXT] No relevant documents found.\n")
            formatted_context = "NO_DATA_FOUND"

        current_time = datetime.datetime.now().strftime("%A, %I:%M %p")
        
        prompt = f"""[INST] You are Bearnard, the AI Concierge of iACADEMY (The Nexus), You are located at the Ground Floor - Lobby. 
Current Time: {current_time}

### INSTRUCTIONS:
1. **SOURCE OF TRUTH:** Answer questions using ONLY the information in the [CONTEXT] block below. Check for slang words or abbrevations used in iACADEMY. (CR for Comfort Room, CL for Computer Lab, etc.). check for lower case of the abreveations as well. the CONTEXT is your only source of truth. you must base your answers SOLELY on that information.
2. **UNKNOWN INFO:** If the [CONTEXT] contains "NO_DATA_FOUND", say: "I'm sorry, I don't have that information in my current records." if the [CONTEXT] doesn't make sense or logical, answer based on your knowledge regarding the CONTEXT. Make sure to analyze the CONTEXT properly and follows the appropriate questions. avoid making up answers. This doesn't apply on Special Rules.
3. **OFF-TOPIC:** If the user asks about math, coding, or general world trivia (not related to iACADEMY), politely decline.
4. **VOICE OPTIMIZATION:** You are speaking to the user.
   - Keep answers **short** (under 2 sentences if possible).
   - Do NOT use lists, bullet points, or markdown formatting.
   - If listing items, separate them with commas for natural speech.

   SPECIAL RULES:
- If asked about NEAREST location, answer based on your location at Ground Floor - Lobby.
- If asked for actions (greet, say hello), respond with a short greeting only.
- If asked for the time, respond with the current time only.

### [CONTEXT]
{formatted_context}

### [USER QUESTION]
{user_text}

### [BEARNARD'S ANSWER]
[/INST]"""

        # Dynamic token limit helps prevents cutting off lists if needed, otherwise 512 is good default
        token_limit = 1024 if "list" in user_text.lower() else 512
        
        answer = self.llm.ask(prompt, max_tokens=token_limit)
        self.response_ready.emit(answer)
        self.state_changed.emit("SPEAKING")
        self.mouth.speak(answer)
        self.state_changed.emit("IDLE")

# ANIMATED BEAR WIDGET
class BearAvatar(QLabel):
    def __init__(self, width=400, height=400):
        super().__init__()
        self.setFixedSize(width, height)
        self.setScaledContents(True)
        self.setStyleSheet("background-color: transparent;")
        
        # PLACEHOLDERS (Replace with your actual images)
        self.img_closed = QPixmap(width, height)
        self.img_closed.fill(QColor("transparent")) 
        
        self.img_open = QPixmap(width, height)
        self.img_open.fill(QColor("transparent"))

        self.setPixmap(self.img_closed)
        self.talk_timer = QTimer()
        self.talk_timer.timeout.connect(self.toggle_mouth)
        self.is_mouth_open = False

    def toggle_mouth(self):
        self.is_mouth_open = not self.is_mouth_open
        self.setPixmap(self.img_open if self.is_mouth_open else self.img_closed)

    def set_state(self, state):
        if state == "SPEAKING":
            if not self.talk_timer.isActive(): self.talk_timer.start(150)
        else:
            self.talk_timer.stop()
            self.setPixmap(self.img_closed)

# CHAT WINDOW (The Dashboard)
class ChatWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Bearnard - Chat Mode")
        self.resize(1100, 650)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout: Horizontal Split
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # LEFT COLUMN (Bear & Title)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_hey = QLabel("HEY THERE! I'M")
        lbl_hey.setStyleSheet("color: white; font-size: 24px; font-weight: bold; font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;")
        lbl_hey.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(lbl_hey)

        # MODIFIED: REPLACED TEXT WITH ASSET IMAGE 
        lbl_name = QLabel()
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Load the Bearnard Text Asset
        # Ensure 'assets/bearnard_text.png' exists in your project root
        logo_pixmap = QPixmap("assets/bearnard_text.png")
        
        if not logo_pixmap.isNull():
            # Scale it to fit nicely (adjust height '80' as needed)
            lbl_name.setPixmap(logo_pixmap.scaledToHeight(50, Qt.TransformationMode.SmoothTransformation))
        else:
            # Fallback if image is missing
            lbl_name.setText("BEARNARD")
            lbl_name.setStyleSheet("color: #3b82f6; font-size: 54px; font-weight: 900; font-family: 'Impact', sans-serif;")
        
        # Keep the shadow effect if you want it on the image too (optional)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(2, 2)
        lbl_name.setGraphicsEffect(shadow)
        
        left_layout.addWidget(lbl_name)

        # Bear Image
        self.bear = BearAvatar(400, 400)
        left_layout.addWidget(self.bear, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(left_col, 35) # 35% Width

        # RIGHT COLUMN (Interaction)
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel("How can I help you today?")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: 500;")
        header.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(header)

        # Buttons Row
        btn_row = QHBoxLayout()
        
        # Button 1: Voice
        self.btn_voice = QPushButton("   Let's Talk!")
        self.btn_voice.setProperty("class", "ModeBtn")
        self.btn_voice.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.btn_voice.setIcon(QIcon("assets/microphone.png"))
        self.btn_voice.setIconSize(QSize(36, 36)) 
        
        self.btn_voice.clicked.connect(lambda: self.controller.set_mode("voice"))
        
        # Button 2: Chat
        self.btn_chat = QPushButton("   Chat with me!")
        self.btn_chat.setProperty("class", "ModeBtn")
        self.btn_chat.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.btn_chat.setIcon(QIcon("assets/chat.png"))
        self.btn_chat.setIconSize(QSize(36, 36))
        
        self.btn_chat.clicked.connect(lambda: self.controller.set_mode("chat"))

        btn_row.addWidget(self.btn_voice)
        btn_row.addWidget(self.btn_chat)
        right_layout.addLayout(btn_row)

        # WHITE CHAT CARD
        card = QFrame()
        card.setObjectName("ChatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        # Chat History
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_content = QWidget()
        self.chat_content.setObjectName("ChatContent")
        self.msg_layout = QVBoxLayout(self.chat_content)
        self.msg_layout.addStretch()
        self.scroll.setWidget(self.chat_content)
        card_layout.addWidget(self.scroll)

        # (Dark Blue bar inside White Card) 
        input_pill = QFrame()
        input_pill.setObjectName("InputPill")
        input_pill.setFixedHeight(50)
        pill_layout = QHBoxLayout(input_pill)
        pill_layout.setContentsMargins(15, 0, 5, 0)

        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Ask anything...")
        self.txt_input.returnPressed.connect(self.send_text)
        
        self.btn_send = QPushButton("⬆") # Arrow Icon
        self.btn_send.setFixedSize(35, 35)
        self.btn_send.setStyleSheet("background-color: white; color: #020621; border-radius: 17px; font-weight: bold;")
        self.btn_send.clicked.connect(self.send_text)

        pill_layout.addWidget(self.txt_input)
        pill_layout.addWidget(self.btn_send)
        
        card_layout.addWidget(input_pill)
        
        right_layout.addWidget(card)
        main_layout.addWidget(right_col, 65) 

    def add_message(self, sender, text):
        msg_lbl = QLabel(f"<b>{sender}:</b> {text}")
        msg_lbl.setProperty("class", "ChatMessage")
        msg_lbl.setWordWrap(True)
        self.msg_layout.addWidget(msg_lbl)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def send_text(self):
        text = self.txt_input.text().strip()
        if not text: return
        self.add_message("You", text)
        self.txt_input.clear()
        self.controller.worker.process_text(text)

    def update_ui_state(self, state):
        self.bear.set_state(state)

# VOICE WINDOW (Second Monitor)
class VoiceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bearnard - Voice Avatar")
        self.resize(600, 900)
        
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(central)

        self.bear = BearAvatar(500, 500)
        layout.addWidget(self.bear, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_status = QLabel("Listening...")
        self.lbl_status.setStyleSheet("font-size: 24px; color: white; opacity: 0.7;")
        layout.addWidget(self.lbl_status, alignment=Qt.AlignmentFlag.AlignCenter)

    def update_ui_state(self, state):
        self.bear.set_state(state)
        self.lbl_status.setText(state)

# MAIN CONTROLLER
class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(STYLESHEET)

        self.worker = AIWorker()
        self.worker.start()

        # Init Windows
        self.chat_window = ChatWindow(self)
        self.voice_window = VoiceWindow()

        # Connect Signals
        self.worker.state_changed.connect(self.chat_window.update_ui_state)
        self.worker.state_changed.connect(self.voice_window.update_ui_state)
        self.worker.response_ready.connect(lambda t: self.chat_window.add_message("Bearnard", t))

        # Show Both Windows
        self.chat_window.show()
        self.voice_window.show()

        sys.exit(self.app.exec())

    def set_mode(self, mode):
        self.worker.set_mode(mode)

if __name__ == "__main__":
    MainController()