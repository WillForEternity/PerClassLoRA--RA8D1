import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame, QStackedWidget
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import subprocess
import os

class TrainingWorker(QThread):
    """Runs the training script as a subprocess in its own venv."""
    new_log_message = pyqtSignal(str)

    def run(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        training_venv_python = os.path.join(project_root, "Python_Hand_Tracker", "venv_training", "bin", "python")
        training_script = os.path.join(project_root, "Python_Hand_Tracker", "train_model.py")

        if not os.path.exists(training_venv_python) or not os.path.exists(training_script):
            self.new_log_message.emit("[ERROR] Training environment or script not found.")
            self.new_log_message.emit(f"Venv Path: {training_venv_python}")
            self.new_log_message.emit(f"Script Path: {training_script}")
            return

        try:
            process = subprocess.Popen(
                [training_venv_python, training_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in iter(process.stdout.readline, ''):
                self.new_log_message.emit(line.strip())
            
            process.stdout.close()
            process.wait()

        except Exception as e:
            self.new_log_message.emit(f"\n[FATAL ERROR] Failed to run training subprocess: {e}")

class TrainingPage(QWidget):
    set_navigation_enabled = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.is_setup_complete = False
        self.training_worker = TrainingWorker()
        self.training_worker.new_log_message.connect(self.append_log_message)
        self.training_worker.finished.connect(self.on_training_finished)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.page_stack = QStackedWidget()
        main_layout.addWidget(self.page_stack)

        # --- Prerequisites View ---
        prereq_view = QWidget()
        prereq_layout = QVBoxLayout(prereq_view)
        prereq_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prereq_label = QLabel("Please complete the environment setup on the 'Setup' page first.")
        prereq_label.setFont(QFont("Arial", 16))
        prereq_layout.addWidget(prereq_label)
        self.page_stack.addWidget(prereq_view)

        # --- Main Content View ---
        main_view = QWidget()
        layout = QVBoxLayout(main_view)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Model Training")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        main_content_layout = QHBoxLayout()
        layout.addLayout(main_content_layout)

        left_panel = QVBoxLayout()
        main_content_layout.addLayout(left_panel, 1)

        self.run_button = QPushButton("Start Training")
        self.run_button.setFont(QFont("Arial", 12))
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.start_training_process)
        left_panel.addWidget(self.run_button)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier New", 10))
        self.log_console.setStyleSheet("background-color: #000; color: #0f0; border: 1px solid #444;")
        self.log_console.setText("Training logs will appear here...")
        left_panel.addWidget(self.log_console)

        graph_placeholder = QLabel("Training Graph (Coming Soon)")
        graph_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_placeholder.setFont(QFont("Arial", 16))
        graph_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        graph_placeholder.setStyleSheet("background-color: #2d2d2d; border: 1px solid #444;")
        main_content_layout.addWidget(graph_placeholder, 1)
        
        self.page_stack.addWidget(main_view)

    def start_training_process(self):
        self.log_console.clear()
        self.run_button.setEnabled(False)
        self.run_button.setText("Training in Progress...")
        self.set_navigation_enabled.emit(False)
        self.training_worker.start()

    @pyqtSlot(str)
    def append_log_message(self, message):
        self.log_console.append(message)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def on_training_finished(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("Start Training")
        self.set_navigation_enabled.emit(True)
        self.append_log_message("\n--- Process Finished ---")

    def set_setup_status(self, is_complete):
        self.is_setup_complete = is_complete
        if self.is_setup_complete:
            self.page_stack.setCurrentIndex(1)
        else:
            self.page_stack.setCurrentIndex(0)
