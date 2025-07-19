import cv2
import numpy as np
import os
import time
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QProcess, QTimer

from gui_app.logic import HandTracker, GesturePredictor, SIM_DIR, MODELS_DIR

class InferenceWorker(QThread):
    """Worker for camera input and gesture prediction."""
    new_frame = pyqtSignal(np.ndarray)
    new_prediction = pyqtSignal(str, float)

    def __init__(self, hand_tracker, gesture_predictor):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.gesture_predictor = gesture_predictor
        self._running = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Get hand landmarks and annotated frame
            hand_landmarks, annotated_frame = self.hand_tracker.process_frame(frame)

            # Get landmark data (None if no hand)
            landmark_data = self.hand_tracker.get_landmark_data(hand_landmarks)

            # Predictor handles None case
            predicted_gesture, confidence = self.gesture_predictor.predict(landmark_data)
            self.new_prediction.emit(predicted_gesture, confidence)

            # Update video feed
            self.new_frame.emit(cv2.flip(annotated_frame, 1).copy())
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class InferencePage(QWidget):
    set_navigation_enabled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hand_tracker = HandTracker()
        self.worker = None
        self.gesture_predictor = None
        self.inference_process = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Real-Time Gesture Inference")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        content_layout = QHBoxLayout()
        self.video_feed = QLabel("Camera Starting...")
        self.video_feed.setFixedSize(640, 480)
        self.video_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_feed.setStyleSheet("background-color: #000; border: 1px solid #444;")
        content_layout.addWidget(self.video_feed)

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Model Selector
        info_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Base FP32 Model", "Quantized INT8 Model"])
        info_layout.addWidget(self.model_selector)
        
        self.prediction_label = QLabel("Prediction: --")
        self.prediction_label.setFont(QFont("Arial", 18))
        info_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 18))
        info_layout.addWidget(self.confidence_label)

        self.start_button = QPushButton("Start Inference")
        self.start_button.clicked.connect(self.toggle_inference)
        info_layout.addWidget(self.start_button)

        # Server Output Console
        info_layout.addWidget(QLabel("Server Output:"))
        self.server_output = QTextEdit()
        self.server_output.setReadOnly(True)
        self.server_output.setFixedHeight(100)
        self.server_output.setStyleSheet("background-color: #2E2E2E; color: #F2F2F2; font-family: 'Courier New';")
        info_layout.addWidget(self.server_output)

        content_layout.addLayout(info_layout)
        layout.addLayout(content_layout)

    def _start_inference(self):
        self.server_output.clear()
        selected_model_text = self.model_selector.currentText()

        if selected_model_text == "Base FP32 Model":
            model_file = "c_model.bin"
            model_path = os.path.join(MODELS_DIR, model_file)
        else:  # Quantized INT8 Model
            model_file = "c_model_quantized.bin"
            model_path = os.path.join(SIM_DIR, model_file)

        if not os.path.exists(model_path):
            self.server_output.setText(f"Error: Model file not found at {model_path}. Please train and/or quantize a model first.")
            return

        self.inference_process = QProcess()
        self.inference_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.inference_process.readyReadStandardOutput.connect(self.on_server_output)
        self.inference_process.finished.connect(self.on_server_finished)

        executable = os.path.join(SIM_DIR, "ra8d1_sim")
        self.server_output.append(f"Starting server with {model_file}...\n")
        self.inference_process.start(executable, [model_path])

        # Give server time to start before connecting
        QTimer.singleShot(1000, self.connect_to_server)

    def connect_to_server(self):
        try:
            self.gesture_predictor = GesturePredictor()
            # Check if connection was successful in GesturePredictor's __init__
            if not self.gesture_predictor.client_socket:
                raise ConnectionRefusedError("Failed to connect to the C server.")

            self.worker = InferenceWorker(self.hand_tracker, self.gesture_predictor)
            self.worker.new_frame.connect(self.update_video_feed)
            self.worker.new_prediction.connect(self.update_prediction)
            self.worker.start()

            self.start_button.setText("Stop Inference")
            self.set_navigation_enabled.emit(False)
            self.model_selector.setEnabled(False)

        except Exception as e:
            self.server_output.append(f"\nError starting inference client: {e}")
            self._stop_inference() # Clean up if connection fails

    def _stop_inference(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        if self.gesture_predictor:
            self.gesture_predictor.cleanup()

        if self.inference_process and self.inference_process.state() == QProcess.ProcessState.Running:
            self.inference_process.terminate()
            self.inference_process.waitForFinished(1000)

        self.worker = None
        self.gesture_predictor = None
        self.inference_process = None

        self.start_button.setText("Start Inference")
        self.set_navigation_enabled.emit(True)
        self.model_selector.setEnabled(True)
        self.video_feed.setText("Camera Stopped")
        self.prediction_label.setText("Prediction: --")
        self.confidence_label.setText("Confidence: --")

    def on_server_output(self):
        output = self.inference_process.readAllStandardOutput().data().decode().strip()
        self.server_output.append(output)

    def on_server_finished(self):
        self.server_output.append("\nInference server stopped.")
        # Ensure UI is reset if server stops unexpectedly
        if self.worker and self.worker.isRunning():
            self._stop_inference()

    def toggle_inference(self):
        if self.worker and self.worker.isRunning():
            self._stop_inference()
        else:
            self._start_inference()

    @pyqtSlot(np.ndarray)
    def update_video_feed(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        # Scale the image to fit the label, maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_feed.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_feed.setPixmap(scaled_pixmap)

    @pyqtSlot(str, float)
    def update_prediction(self, gesture, confidence):
        if gesture == "No Hand Present":
            self.prediction_label.setText("Prediction: No Hand")
            self.confidence_label.setText("Confidence: --")
        elif gesture == "Collecting data...":
            self.prediction_label.setText("Status: Collecting...")
            self.confidence_label.setText("Confidence: --")
        else:
            self.prediction_label.setText(f"Prediction: {gesture.capitalize()}")
            self.confidence_label.setText(f"Confidence: {confidence:.2f}")

    def showEvent(self, event):
        super().showEvent(event)
        # Auto-start inference on show
        if not self.worker or not self.worker.isRunning():
            self._start_inference()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Auto-stop inference on hide
        if self.worker and self.worker.isRunning():
            self._stop_inference()
