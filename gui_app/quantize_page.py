from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QScrollArea
from PyQt6.QtCore import pyqtSignal
from .logic import Quantizer

class QuantizePage(QWidget):
    set_navigation_enabled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.quantizer = Quantizer()

        layout = QVBoxLayout(self)
        self.setWindowTitle("Model Quantization")

        # --- Title and Description ---
        title_label = QLabel("Post-Training Quantization (PTQ)")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        description = QLabel(
            "This process converts the trained floating-point model (FP32) into a smaller, more efficient \n"
            "8-bit integer model (INT8). This reduces the model's size and can significantly speed up \n"
            "inference, which is crucial for deployment on resource-constrained devices like microcontrollers."
        )
        description.setWordWrap(True)

        # --- Quantize Button ---
        self.quantize_button = QPushButton("Quantize Trained Model")
        self.quantize_button.clicked.connect(self.run_quantization)

        # --- Output Console ---
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("background-color: #2E2E2E; color: #F2F2F2; font-family: 'Courier New';")

        # --- Add widgets to layout ---
        layout.addWidget(title_label)
        layout.addWidget(description)
        layout.addWidget(self.quantize_button)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_console)

        # Connect signals from the quantizer logic
        self.quantizer.output_received.connect(self.append_output)
        self.quantizer.quantization_finished.connect(self.on_finished)

    def run_quantization(self):
        self.output_console.clear()
        self.append_output("Starting quantization process...")
        self.quantize_button.setEnabled(False)
        self.quantizer.run_quantization()

    def append_output(self, text):
        self.output_console.append(text)
        self.output_console.verticalScrollBar().setValue(self.output_console.verticalScrollBar().maximum())

    def on_finished(self, exit_code):
        if exit_code == 0:
            self.append_output("\nQuantization completed successfully!")
        else:
            self.append_output(f"\nQuantization failed with exit code: {exit_code}")
        self.quantize_button.setEnabled(True)
