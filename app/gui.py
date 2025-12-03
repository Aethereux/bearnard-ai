import sys
import time
import datetime
import platform
import torch
import sounddevice as sd
import traceback 

# --- ENGINE IMPORTS ---
from faster_whisper import WhisperModel

# Graceful MLX Import
HAS_MLX = False
try:
    if platform.system() == "Darwin":
        from mlx_wrapper import MLXWhisperWrapper
        import mlx_whisper 
        HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    pass

from rag import Rag
from llm import LLM
from voice_input import VoiceInput
from voice_output import VoiceOutput
from wake_word import WakeWordDetector

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QScrollArea, QFrame, QGraphicsDropShadowEffect,
                             QDialog, QComboBox, QDialogButtonBox, QProgressBar, QTextEdit,
                             QStackedLayout) # New Import: QStackedLayout
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QColor, QIcon, QResizeEvent # New Import: QResizeEvent

# --- VISUAL CONSTANTS ---
WHITE_PANEL = "#ffffff"     
INPUT_BG = "#1b1b3a"        
BUTTON_BG = "#1f1f1f"       
ACTIVE_BTN_BG = "#3b82f6"   
TEXT_COLOR = "#333333"      

STYLESHEET = f"""
    QMainWindow {{
        background: qradialgradient(
            spread:pad, cx:0, cy:0, radius:1.8, fx:0, fy:0,
            stop:0 #000b21, 
            stop:0.5 #081378, 
            stop:1 #1423a8
        );
    }}
    QWidget {{ font-family: 'Segoe UI', sans-serif; }}
    
    #ChatCard {{ background-color: {WHITE_PANEL}; border-radius: 20px; }}
    
    QPushButton.ModeBtn {{
        background-color: {BUTTON_BG}; color: white; border-radius: 15px;
        padding: 5px 20px; font-weight: bold; font-size: 15px; text-align: left;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    QPushButton.ModeBtn:hover {{ background-color: #333333; }}

    QPushButton#MicBtn {{
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 40px;
        color: white; font-weight: bold; font-size: 16px;
    }}
    QPushButton#MicBtn:hover {{
        background-color: rgba(255, 255, 255, 0.2);
        border: 2px solid white;
    }}
    QPushButton#MicBtn:pressed {{
        background-color: rgba(59, 130, 246, 0.5); 
    }}

    QLabel#StatusLabel {{
        color: #00ff00;
        font-weight: bold;
        font-size: 14px;
        background-color: rgba(0, 0, 0, 0.4);
        padding: 6px 12px;
        border-radius: 6px;
    }}
    
    QLabel#ModeLabel {{
        font-weight: 900;
        font-size: 14px;
        padding: 8px;
        border-radius: 8px;
        text-transform: uppercase;
        margin-top: 10px;
        margin-bottom: 5px;
    }}

    QScrollArea {{ border: none; background-color: transparent; }}
    QWidget#ChatContent {{ background-color: transparent; }}
    QLabel.ChatMessage {{ color: {TEXT_COLOR}; font-size: 14px; padding: 5px; }}

    QFrame#InputPill {{ background-color: {INPUT_BG}; border-radius: 25px; }}
    QLineEdit {{ background-color: transparent; color: white; font-size: 14px; border: none; }}
    QLineEdit::placeholder {{ color: #aaaaaa; }}

    QProgressBar {{ border: 2px solid white; border-radius: 5px; text-align: center; background: #000b21; }}
    QProgressBar::chunk {{ background-color: #3b82f6; }}
    
    QDialog {{ background-color: #081378; color: white; }}
    QComboBox {{ padding: 5px; border-radius: 5px; background-color: white; color: black; }}
"""

# --- 1. MICROPHONE SELECTION DIALOG (UNCHANGED) ---
class DeviceSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Microphone")
        self.setFixedSize(400, 150)
        self.selected_index = None
        layout = QVBoxLayout(self)
        lbl = QLabel("Please select your microphone:")
        lbl.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(lbl)
        self.combo = QComboBox()
        self.devices = []
        try:
            all_devices = sd.query_devices()
            for i, dev in enumerate(all_devices):
                if dev['max_input_channels'] > 0:
                    self.devices.append((i, dev['name']))
                    self.combo.addItem(f"{dev['name']} (ID: {i})")
        except Exception as e:
            self.combo.addItem(f"Error: {e}")
        layout.addWidget(self.combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def accept_selection(self):
        if self.devices:
            idx = self.combo.currentIndex()
            self.selected_index = self.devices[idx][0]
        self.accept()

# --- 2. TRANSCRIPT LOG WINDOW (UNCHANGED) ---
class TranscriptWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Transcript Log")
        self.resize(500, 400)
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QTextEdit { 
                background-color: #0d1117; 
                color: #00ff00; 
                font-family: 'Consolas', 'Courier New', monospace; 
                font-size: 12px;
                border: none;
                padding: 10px;
            }
        """)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.setCentralWidget(self.log_area)
        self.last_log = "" 
        self.log("--- SYSTEM INITIALIZED ---")

    def log(self, text, prefix="INFO"):
        if text == self.last_log and prefix == "HEARD":
            return
            
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{prefix}] {text}"
        self.log_area.append(formatted)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.last_log = text

# --- 3. AI WORKER (UNCHANGED) ---
class AIWorker(QThread):
    response_ready = pyqtSignal(str)
    state_changed = pyqtSignal(str)
    mic_volume = pyqtSignal(int)
    transcribed_text = pyqtSignal(str)
    log_message = pyqtSignal(str, str)
    
    def __init__(self, mic_index=None):
        super().__init__()
        self.mode = "chat"
        self.is_running = True
        self.input_queue = None
        self.mic_index = mic_index 
        self.manual_trigger_active = False 

    def set_mode(self, mode):
        self.mode = mode
        self.log_message.emit(f"Switched to {mode.upper()} mode.", "MODE")

    def trigger_wake(self):
        if self.mode == "voice":
            self.manual_trigger_active = True
            self.log_message.emit("Manual 'Tap to Speak' triggered.", "INPUT")

    def process_text(self, text):
        self.input_queue = text

    def run(self):
        try:
            self.state_changed.emit("LOADING")
            print(f"⏳ Loading Models... (Mic Index: {self.mic_index})")
            
            # --- ENGINE SELECTION LOGIC ---
            os_name = platform.system()
            shared_whisper = None
            
            if os_name == "Darwin" and HAS_MLX:
                msg = "🍎 Apple Silicon (M4) Detected. Using MLX Engine."
                print(msg)
                self.log_message.emit(msg, "SYS")
                shared_whisper = MLXWhisperWrapper("mlx-community/whisper-large-v3-turbo")
                
            elif torch.cuda.is_available():
                msg = "🚀 NVIDIA GPU Detected. Using Faster-Whisper (Float16)."
                print(msg)
                self.log_message.emit(msg, "SYS")
                shared_whisper = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
                
            else:
                msg = "💻 Standard CPU Detected. Using Faster-Whisper (Int8)."
                print(msg)
                self.log_message.emit(msg, "SYS")
                shared_whisper = WhisperModel("distil-medium.en", device="cpu", compute_type="int8", cpu_threads=4)

            self.rag = Rag(build_if_empty=True)
            try:
                self.llm = LLM()
            except Exception as e:
                self.log_message.emit(f"LLM Error: {e}", "ERROR")
                self.llm = None 

            self.ear = VoiceInput(model=shared_whisper, device=self.mic_index)
            self.wake = WakeWordDetector(model=shared_whisper, device=self.mic_index)
            self.mouth = VoiceOutput()
            
            # --- CRITICAL CALIBRATION ---
            self.state_changed.emit("CALIBRATING")
            self.log_message.emit("Calibrating Microphone...", "CALIB")
            self.ear.adjust_for_ambient_noise()
            
            # SYNC THRESHOLDS: Apply Ear's calculated threshold to Wake Word
            # This prevents the wake word from listening to silence/static
            self.wake.energy_threshold = self.ear.silence_threshold
            self.log_message.emit(f"VAD Threshold Synced: {self.wake.energy_threshold:.4f}", "SYS")

            self.state_changed.emit("IDLE")
            self.log_message.emit("AI Ready.", "READY")

            while self.is_running:
                if self.input_queue:
                    text = self.input_queue
                    self.input_queue = None
                    self.log_message.emit(f"User typed: {text}", "CHAT")
                    self.generate_response(text)
                    continue

                if self.mode == "voice":
                    wake_heard = self.wake.listen_for_wake_word(
                        timeout=0.1, 
                        volume_callback=lambda vol: self.mic_volume.emit(int(vol * 500)),
                        transcript_callback=lambda text: self.log_message.emit(f"'{text}'", "HEARD")
                    )
                    
                    if wake_heard or self.manual_trigger_active:
                        self.wake.stop_stream()
                        
                        if wake_heard:
                            self.log_message.emit("Wake Word Detected!", "WAKE")
                        
                        self.manual_trigger_active = False 
                        
                        self.state_changed.emit("LISTENING")
                        self.mic_volume.emit(50) 
                        self.log_message.emit("Recording...", "REC")
                        
                        audio = self.ear.record_until_silence(
                            callback=lambda vol: self.mic_volume.emit(int(vol * 500))
                        )
                        
                        self.state_changed.emit("THINKING")
                        self.mic_volume.emit(0) 
                        self.log_message.emit("Transcribing...", "PROC")
                        
                        user_text = self.ear.transcribe(audio)
                        
                        if user_text.strip():
                            self.log_message.emit(f"Heard: '{user_text}'", "VOICE")
                            self.transcribed_text.emit(user_text)
                            self.generate_response(user_text)
                        else:
                            self.log_message.emit("Heard nothing.", "INFO")
                            self.state_changed.emit("IDLE")
                
                else:
                    time.sleep(0.1)
        
        except Exception as e:
            err_msg = f"CRASH: {str(e)}"
            print(err_msg)
            traceback.print_exc()
            self.log_message.emit(err_msg, "CRITICAL")
            self.state_changed.emit("IDLE")

    def generate_response(self, user_text):
        self.state_changed.emit("THINKING")
        
        # --- MODIFICATION 1: FORCE DEEPER SEARCH ---
        # We pass n_results=15 to override the default 5 in rag.py.
        # This ensures we find "Clinic = Ground Floor" even if it's ranked #10.
        docs = self.rag.search(user_text, n_results=15)
        
        # Log context results (Your original logic)
        if docs:
            print(f"\n📚 [CONTEXT] Found {len(docs)} relevant documents:")
            for i, doc in enumerate(docs, 1):
                # Added replace('\n', ' ') just to keep your console clean
                print(f"  [{i}] {doc[:100].replace('\n', ' ')}..." if len(doc) > 100 else f"  [{i}] {doc}")
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
2. **UNKNOWN INFO:** If the [CONTEXT] contains "NO_DATA_FOUND", say: "I'm sorry, I don't have that information in my current records." or If the [CONTEXT] doesn't make sense or logical, answer based on your knowledge regarding the CONTEXT. Make sure to analyze the CONTEXT properly and follows the appropriate questions. avoid making up answers. This doesn't apply on Special Rules.
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
        
        # --- MODIFICATION 2: ECHO CANCELLATION ---
        # Calculate how long the AI takes to speak (approx 0.3s/word) + 2.0s safety buffer.
        # This pauses the thread so the mic stays OFF while the AI is talking.
        word_count = len(answer.split())
        wait_time = (word_count * 0.3) + 2.0
        
        # Optional: Print log so you know it's waiting
        # print(f"⏳ Cooling down for {wait_time:.1f}s...")
        time.sleep(wait_time)
        
        # --- MODIFICATION 3: WIPE MEMORY ---
        # Clear the wake word buffer immediately so it doesn't process the echo it just heard.
        if hasattr(self.wake, 'audio_buffer'):
            self.wake.audio_buffer.clear()

        self.state_changed.emit("IDLE")

# --- NEW: STATIC BEAR AVATAR (For Chat Window) ---
class StaticBearAvatar(QLabel):
    def __init__(self, width=450, height=450):
        super().__init__()
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: transparent;")
        
        # NOTE: Assuming 'assets/bearnard_chat_static.png' exists.
        img_static = QPixmap("assets/bearnard_chat_static.png")
        
        if not img_static.isNull():
            # Scale for high quality view, keeping aspect ratio
            img_scaled = img_static.scaled(width, height, 
                                           Qt.AspectRatioMode.KeepAspectRatio, 
                                           Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(img_scaled)
        else:
             # Fallback if image not found
            self.setText("🐻 Static Bearnard")
            self.setStyleSheet("color: white; font-size: 18px;")

    def set_state(self, state):
        # Static Bearnard does not animate
        pass

# --- RENAMED & MODIFIED: ANIMATED BEAR AVATAR (For Voice Window Background) ---
class AnimatedBearAvatar(QLabel):
    def __init__(self):
        super().__init__()
        # IMPORTANT: Remove setFixedSize to allow dynamic resizing
        # self.setFixedSize(width, height) 
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        
        # 1. Load the original image files (keep original pixmaps)
        self.img_closed_original = QPixmap("assets/bearnard_closedMouth.png")
        self.img_open_original = QPixmap("assets/bearnard_openMouth.png")
        
        # Initialize scaled pixmaps
        self.img_closed = self.img_closed_original
        self.img_open = self.img_open_original
        
        # Set initial image
        self.setPixmap(self.img_closed)
        
        self.talk_timer = QTimer()
        self.talk_timer.timeout.connect(self.toggle_mouth)
        self.is_mouth_open = False

    def resize_images(self, size):
        """Scales the pixmaps without stretching, based on the new size."""
        
        # Use a large size for scaling to ensure it fills the background
        # Qt.AspectRatioMode.KeepAspectRatioByExpanding ensures the image covers the entire area
        # while Qt.TransformationMode.SmoothTransformation maintains quality.
        if not self.img_closed_original.isNull():
            self.img_closed = self.img_closed_original.scaled(
                size, 
                Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
                Qt.TransformationMode.SmoothTransformation
            )
        if not self.img_open_original.isNull():
            self.img_open = self.img_open_original.scaled(
                size, 
                Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
                Qt.TransformationMode.SmoothTransformation
            )
        
        # Re-set the current pixmap to apply the scaling
        current_img = self.img_open if self.is_mouth_open else self.img_closed
        self.setPixmap(current_img)

    def resizeEvent(self, event: QResizeEvent):
        """Called when the widget is resized. Triggers image rescaling."""
        self.resize_images(event.size())
        super().resizeEvent(event)
        
    def toggle_mouth(self):
        self.is_mouth_open = not self.is_mouth_open
        self.setPixmap(self.img_open if self.is_mouth_open else self.img_closed)
        
    def set_state(self, state):
        if state == "SPEAKING":
            # Start the mouth-toggle animation
            if not self.talk_timer.isActive(): self.talk_timer.start(150)
        else:
            # Stop the animation and revert to the closed mouth
            self.talk_timer.stop()
            self.setPixmap(self.img_closed)

# --- 4. CHAT WINDOW (UNCHANGED) ---
class ChatWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Bearnard - Chat Mode")
        self.resize(1100, 650)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        lbl_hey = QLabel("HEY THERE! I'M")
        lbl_hey.setStyleSheet("color: white; font-size: 24px; font-weight: bold; font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;")
        lbl_hey.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(lbl_hey)
        lbl_name = QLabel()
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_pixmap = QPixmap("assets/bearnard_text.png")
        if not logo_pixmap.isNull():
            lbl_name.setPixmap(logo_pixmap.scaledToHeight(50, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_name.setText("BEARNARD")
            lbl_name.setStyleSheet("color: #3b82f6; font-size: 54px; font-weight: 900; font-family: 'Impact', sans-serif;")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(2, 2)
        lbl_name.setGraphicsEffect(shadow)
        left_layout.addWidget(lbl_name)
        
        # --- MODIFICATION: Use StaticBearAvatar ---
        self.bear = StaticBearAvatar() 
        left_layout.addWidget(self.bear, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_mode = QLabel("CURRENT MODE: ⌨️ CHAT")
        self.lbl_mode.setObjectName("ModeLabel")
        self.lbl_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_mode)
        self.lbl_system_status = QLabel("⚫ System Idle")
        self.lbl_system_status.setObjectName("StatusLabel")
        self.lbl_system_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_system_status)
        self.vol_bar = QProgressBar()
        self.vol_bar.setRange(0, 50)
        self.vol_bar.setFixedHeight(8)
        self.vol_bar.setTextVisible(False)
        left_layout.addWidget(self.vol_bar)
        main_layout.addWidget(left_col, 35) 
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(10, 10, 10, 10)
        header = QLabel("How can I help you today?")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: 500;")
        header.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(header)
        btn_row = QHBoxLayout()
        self.btn_voice = QPushButton("   Let's Talk!")
        self.btn_voice.setProperty("class", "ModeBtn")
        self.btn_voice.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_voice.setIcon(QIcon("assets/microphone.png"))
        self.btn_voice.setIconSize(QSize(50, 50)) 
        self.btn_voice.clicked.connect(lambda: self.controller.set_mode("voice"))
        self.btn_chat = QPushButton("   Chat with me!")
        self.btn_chat.setProperty("class", "ModeBtn")
        self.btn_chat.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_chat.setIcon(QIcon("assets/chat.png"))
        self.btn_chat.setIconSize(QSize(50, 50))
        self.btn_chat.clicked.connect(lambda: self.controller.set_mode("chat"))
        btn_row.addWidget(self.btn_voice)
        btn_row.addWidget(self.btn_chat)
        right_layout.addLayout(btn_row)
        card = QFrame()
        card.setObjectName("ChatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_content = QWidget()
        self.chat_content.setObjectName("ChatContent")
        self.msg_layout = QVBoxLayout(self.chat_content)
        self.msg_layout.addStretch()
        self.scroll.setWidget(self.chat_content)
        card_layout.addWidget(self.scroll)
        input_pill = QFrame()
        input_pill.setObjectName("InputPill")
        input_pill.setFixedHeight(50)
        pill_layout = QHBoxLayout(input_pill)
        pill_layout.setContentsMargins(15, 0, 5, 0)
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Ask anything...")
        self.txt_input.returnPressed.connect(self.send_text)
        self.btn_send = QPushButton("⬆")
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
        if state == "IDLE":
            self.lbl_system_status.setText("👂 Waiting for Wake Word...")
            self.lbl_system_status.setStyleSheet("color: #00ff00; background-color: rgba(0,255,0,0.1); font-weight: bold; border-radius: 4px;")
        elif state == "LISTENING":
            self.lbl_system_status.setText("🎙️ Listening to You...")
            self.lbl_system_status.setStyleSheet("color: #3b82f6; background-color: rgba(59,130,246,0.1); font-weight: bold; border-radius: 4px;")
        elif state == "THINKING":
            self.lbl_system_status.setText("🧠 Thinking...")
            self.lbl_system_status.setStyleSheet("color: #eab308; background-color: rgba(234,179,8,0.1); font-weight: bold; border-radius: 4px;")
        elif state == "SPEAKING":
            self.lbl_system_status.setText("🗣️ Speaking...")
            self.lbl_system_status.setStyleSheet("color: #ec4899; background-color: rgba(236,72,153,0.1); font-weight: bold; border-radius: 4px;")
        elif state == "CALIBRATING":
            self.lbl_system_status.setText("🔧 Calibrating Mic...")
    def update_volume(self, level):
        self.vol_bar.setValue(level)
    def set_mode_visuals(self, mode):
        if mode == "voice":
            self.lbl_mode.setText("CURRENT MODE: 🎤 VOICE")
            self.lbl_mode.setStyleSheet("color: #3b82f6; border: 2px solid #3b82f6; background-color: rgba(59,130,246,0.1);")
            self.btn_voice.setStyleSheet(f"background-color: {ACTIVE_BTN_BG}; color: white; border: 2px solid white;")
            self.btn_chat.setStyleSheet(f"background-color: {BUTTON_BG}; color: white; border: 1px solid rgba(255,255,255,0.1);")
        else:
            self.lbl_mode.setText("CURRENT MODE: ⌨️ CHAT")
            self.lbl_mode.setStyleSheet("color: #ffffff; border: 2px solid #ffffff; background-color: rgba(255,255,255,0.1);")
            self.btn_chat.setStyleSheet(f"background-color: {ACTIVE_BTN_BG}; color: white; border: 2px solid white;")
            self.btn_voice.setStyleSheet(f"background-color: {BUTTON_BG}; color: white; border: 1px solid rgba(255,255,255,0.1);")

# --- 5. VOICE WINDOW (MODIFIED for Full-Screen Animated Background) ---
class VoiceWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Bearnard - Voice Mode")
        self.resize(500, 700)
        
        # Remove default QMainWindow background gradient via STYLESHEET
        self.setStyleSheet("QMainWindow { background-color: black; }") 
        
        # 1. Create the Animated Bear which will serve as the background
        self.bear = AnimatedBearAvatar() 
        
        # 2. Create an overlay widget for all other UI elements (status, buttons)
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        overlay_widget.setStyleSheet("background-color: transparent;")
        
        # Add Status Label
        self.lbl_status = QLabel("Initializing...")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.lbl_status)
        
        # Add Spacer to push status and mic button apart
        overlay_layout.addStretch()
        
        # Add the 'Tap to Speak' button, only visible in voice mode (Re-added logic for button from earlier)
        self.mic_btn = QPushButton()
        mic_icon = QIcon("assets/microphone.png")
        self.mic_btn.setIcon(mic_icon)
        self.mic_btn.setIconSize(QSize(60, 60))
        self.mic_btn.setObjectName("MicBtn")
        self.mic_btn.setFixedSize(80, 80)
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_btn.clicked.connect(self.manual_trigger)
        overlay_layout.addWidget(self.mic_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_mic_help = QLabel("Tap to Speak (If Wake Word Fails)")
        self.lbl_mic_help.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_mic_help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.lbl_mic_help)
        
        # 3. Use a QWidget with QStackedLayout to layer the bear and the overlay
        central_widget = QWidget()
        stacked_layout = QStackedLayout(central_widget)
        stacked_layout.setStackingMode(QStackedLayout.StackingMode.StackAll) # Show all layers
        stacked_layout.setContentsMargins(0, 0, 0, 0)
        
        # Layer 1: The animated bear (the background)
        stacked_layout.addWidget(self.bear) 
        
        # Layer 2: The status label and buttons (the foreground)
        stacked_layout.addWidget(overlay_widget) 
        
        self.setCentralWidget(central_widget)

    def manual_trigger(self):
        self.controller.worker.trigger_wake()
        
    def update_ui_state(self, state):
        self.bear.set_state(state)
        if state == "IDLE":
            self.lbl_status.setText("👂 Waiting for Wake Word...")
            # Note: Changed background color for IDLE status for better contrast on black background
            self.lbl_status.setStyleSheet("color: #00ff00; background-color: rgba(0,255,0,0.1); border-radius: 10px; font-size: 16px; padding: 10px;")
        elif state == "LISTENING":
            self.lbl_status.setText("🎙️ I'm Listening!")
            self.lbl_status.setStyleSheet("color: white; background-color: #3b82f6; border-radius: 10px; font-size: 16px; padding: 10px;")
        elif state == "THINKING":
            self.lbl_status.setText("🧠 Thinking...")
            self.lbl_status.setStyleSheet("color: black; background-color: #eab308; border-radius: 10px; font-size: 16px; padding: 10px;")
        elif state == "SPEAKING":
            self.lbl_status.setText("🗣️ Speaking...")
            self.lbl_status.setStyleSheet("color: white; background-color: #ec4899; border-radius: 10px; font-size: 16px; padding: 10px;")

# --- 6. MAIN CONTROLLER (UNCHANGED) ---
class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(STYLESHEET)
        
        # 1. Start Worker First (So it initializes MLX/Model)
        mic_dialog = DeviceSelectionDialog()
        selected_mic_index = mic_dialog.selected_index if mic_dialog.exec() == QDialog.DialogCode.Accepted else None
        print(f"🎤 MainController: Selected Mic Index = {selected_mic_index}")
        
        self.worker = AIWorker(mic_index=selected_mic_index)
        
        # 2. Create Windows
        self.chat_window = ChatWindow(self)
        self.voice_window = VoiceWindow(self)
        self.transcript_window = TranscriptWindow()
        
        # 3. Connect Signals (So logs appear immediately)
        self.worker.state_changed.connect(self.chat_window.update_ui_state)
        self.worker.state_changed.connect(self.voice_window.update_ui_state)
        self.worker.response_ready.connect(lambda t: self.chat_window.add_message("Bearnard", t))
        self.worker.mic_volume.connect(self.chat_window.update_volume)
        self.worker.transcribed_text.connect(lambda t: self.chat_window.add_message("You", t))
        self.worker.log_message.connect(self.transcript_window.log)
        
        # 4. Show & Start
        self.chat_window.show()
        self.voice_window.show()
        self.transcript_window.show()
        self.set_mode("chat")
        
        self.worker.start() # Start processing loop LAST
        
        sys.exit(self.app.exec())

    def set_mode(self, mode):
        self.worker.set_mode(mode)
        self.chat_window.set_mode_visuals(mode)

if __name__ == "__main__":
    MainController()