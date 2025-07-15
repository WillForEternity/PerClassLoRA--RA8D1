import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame, QStackedWidget
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
import subprocess
import os
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """A custom Matplotlib widget for PyQt."""
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
    """Runs the training script as a subprocess in its own venv."""
    new_log_message = pyqtSignal(str)

    def run(self):
        self.new_log_message.emit("--- Initializing training process...")
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

        self.graph_update_timer = QTimer(self)
        self.graph_update_timer.setInterval(2000)  # Refresh every 2 seconds
        self.graph_update_timer.timeout.connect(self.update_graph)

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

        self.graph_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_content_layout.addWidget(self.graph_canvas, 1) # Give graph a stretch factor of 1
        
        self.page_stack.addWidget(main_view)

    def start_training_process(self):
        self.log_console.clear()
        # Clear previous graph
        self.graph_canvas.axes.clear()
        self.graph_canvas.axes.set_title('Training Progress', color='white')
        self.graph_canvas.axes.set_xlabel('Epoch', color='white')
        self.graph_canvas.axes.set_ylabel('Accuracy', color='white')
        self.graph_canvas.draw()

        self.run_button.setEnabled(False)
        self.run_button.setText("Training in Progress...")
        self.set_navigation_enabled.emit(False)
        self.training_worker.start()
        self.graph_update_timer.start()

    @pyqtSlot(str)
    def append_log_message(self, message):
        self.log_console.append(message)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def on_training_finished(self):
        self.graph_update_timer.stop()
        self.run_button.setEnabled(True)
        self.run_button.setText("Start Training")
        self.set_navigation_enabled.emit(True)
        self.append_log_message("\n--- Process Finished ---")
        self.update_graph() # One final update

    def update_graph(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        history_path = os.path.join(project_root, 'models', 'training_history.csv')

        if not os.path.exists(history_path):
            # It's normal for the file not to exist at the very beginning
            return

        try:
            history_df = pd.read_csv(history_path)
            self.graph_canvas.axes.clear() # Clear previous plot
            self.graph_canvas.axes.plot(history_df['accuracy'], label='Training Accuracy', color='cyan')
            self.graph_canvas.axes.plot(history_df['val_accuracy'], label='Validation Accuracy', color='magenta')
            self.graph_canvas.axes.set_title('Training vs. Validation Accuracy', color='white')
            self.graph_canvas.axes.set_xlabel('Epoch', color='white')
            self.graph_canvas.axes.set_ylabel('Accuracy', color='white')
            legend = self.graph_canvas.axes.legend()
            for text in legend.get_texts():
                text.set_color("white")
            self.graph_canvas.draw()
            self.graph_canvas.draw()
            # Do not log every update to avoid spamming the console
            # self.append_log_message("[INFO] Training graph updated.")
        except Exception as e:
            self.append_log_message(f"[GRAPH ERROR] Failed to plot history: {e}")

    def set_setup_status(self, is_complete):
        self.is_setup_complete = is_complete
        if self.is_setup_complete:
            self.page_stack.setCurrentIndex(1)
        else:
            self.page_stack.setCurrentIndex(0)
