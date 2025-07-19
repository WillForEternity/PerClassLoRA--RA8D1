from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QThread, pyqtSignal

from gui_app.logic import run_setup

class SetupWorker(QThread):
    """Worker for the setup process."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def run(self):
        try:
            for line in run_setup():
                self.progress.emit(line)
                if "[ERROR]" in line:
                    raise RuntimeError("Setup script encountered an error.")
            self.finished.emit(True)
        except Exception:
            self.finished.emit(False)


class SetupPage(QWidget):
    setup_completed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Project Setup")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        description = QLabel("Run the initial setup to create virtual environments, install dependencies, and compile the C simulation. This only needs to be run once.")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.run_button = QPushButton("Run Setup")
        self.run_button.setFixedWidth(150)
        self.run_button.clicked.connect(self.start_setup)
        layout.addWidget(self.run_button)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier New", 10))
        self.log_console.setStyleSheet("background-color: #111; color: #ddd; border: 1px solid #444;")
        layout.addWidget(self.log_console)

    def start_setup(self):
        self.run_button.setEnabled(False)
        self.log_console.clear()
        self.log_console.append("Starting setup...\n")
        
        self.worker = SetupWorker()
        self.worker.progress.connect(self.update_console)
        self.worker.finished.connect(self.setup_finished)
        self.worker.start()

    def update_console(self, text):
        self.log_console.append(text.strip())
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def setup_finished(self, success):
        if success:
            self.log_console.append("\nSetup finished successfully.")
        else:
            self.log_console.append("\nSetup failed. See logs for details.")
        self.run_button.setEnabled(True)
        self.setup_completed.emit(success)
