import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame, QStackedWidget
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
import subprocess
import os
import signal
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """Matplotlib widget for PyQt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e1e')
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white') 
        self.axes.spines['right'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')
        self.axes.yaxis.label.set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.title.set_color('white')
        super(MplCanvas, self).__init__(fig)

class TrainingWorker(QThread):
    """Run the C training executable."""
    new_log_message = pyqtSignal(str)

    def run(self):
        self.new_log_message.emit("Initializing C training...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        c_executable_dir = os.path.join(project_root, "RA8D1_Simulation")
        c_executable_path = os.path.join(c_executable_dir, "train_c")

        if not os.path.exists(c_executable_path):
            self.new_log_message.emit(f"Error: C training executable not found at {c_executable_path}")
            self.new_log_message.emit("Please run 'make' in the RA8D1_Simulation directory.")
            return

        try:
            process = subprocess.Popen(
                [c_executable_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=c_executable_dir
            )

            for line in iter(process.stdout.readline, ''):
                self.new_log_message.emit(line.strip())
            
            process.stdout.close()
            process.wait()

        except Exception as e:
            self.new_log_message.emit(f"\nError: Failed to run training: {e}")

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

        # --- Main Content Layout (Vertical) ---
        main_content_layout = QVBoxLayout()
        layout.addLayout(main_content_layout)

        self.run_button = QPushButton("Start Training")
        self.run_button.setFont(QFont("Arial", 12))
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.start_training_process)
        main_content_layout.addWidget(self.run_button)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier New", 10))
        self.log_console.setStyleSheet("background-color: #000; color: #0f0; border: 1px solid #444;")
        self.log_console.setText("Training logs will appear here...")
        main_content_layout.addWidget(self.log_console, 1) # Give log console a stretch factor of 1
        
        self.page_stack.addWidget(main_view)

    def start_training_process(self):
        self.log_console.clear()
        self.run_button.setEnabled(False)
        self.run_button.setText("Training...")
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
        self.append_log_message("\nC training finished.")
        self.restart_c_server()

    def restart_c_server(self):
        self.append_log_message("\n--- Restarting C Inference Server ---")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        pid_file_path = os.path.join(project_root, ".c_server.pid")
        log_file_path = os.path.join(project_root, "c_server.log")
        server_executable_dir = os.path.join(project_root, "RA8D1_Simulation")
        server_executable_path = os.path.join(server_executable_dir, "ra8d1_sim")

        # 1. Stop the existing server
        try:
            if os.path.exists(pid_file_path):
                with open(pid_file_path, 'r') as f:
                    pid = int(f.read().strip())
                self.append_log_message(f"Stopping existing server (PID: {pid})...")
                os.kill(pid, signal.SIGTERM)
                os.remove(pid_file_path)
            else:
                self.append_log_message("No existing server PID file found. Assuming server is not running.")
        except (ProcessLookupError, ValueError, FileNotFoundError) as e:
            self.append_log_message(f"Could not stop previous server (it may have already been stopped): {e}")

        # 2. Start the new server
        try:
            self.append_log_message(f"Starting new server from: {server_executable_path}")
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(
                    [server_executable_path],
                    cwd=server_executable_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
            # 3. Save the new PID
            with open(pid_file_path, 'w') as f:
                f.write(str(process.pid))
            self.append_log_message(f"New C server started successfully (PID: {process.pid}). Logs at c_server.log")
        except Exception as e:
            self.append_log_message(f"Error restarting C server: {e}")

    def set_setup_status(self, is_complete):
        self.is_setup_complete = is_complete
        if self.is_setup_complete:
            self.page_stack.setCurrentIndex(1)
        else:
            self.page_stack.setCurrentIndex(0)
