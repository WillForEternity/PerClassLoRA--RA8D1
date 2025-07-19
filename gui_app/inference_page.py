import cv2
import numpy as np
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from gui_app.logic import HandTracker, GesturePredictor

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
            self.new_frame.emit(cv2.flip(annotated_frame, 1))
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
        
        self.prediction_label = QLabel("Prediction: --")
        self.prediction_label.setFont(QFont("Arial", 18))
        info_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 18))
        info_layout.addWidget(self.confidence_label)

        self.start_button = QPushButton("Start Inference")
        self.start_button.clicked.connect(self.toggle_inference)
        info_layout.addWidget(self.start_button)

        content_layout.addLayout(info_layout)
        layout.addLayout(content_layout)

    def _start_inference(self):
        # Check for model file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        c_model_path = os.path.join(project_root, 'models', 'c_model.bin')
        if not os.path.exists(c_model_path):
            self.prediction_label.setText("Error: c_model.bin not found!")
            self.start_button.setEnabled(False)
            return

        try:
            # 2. Create predictor and worker in correct order
            self.gesture_predictor = GesturePredictor()
            self.worker = InferenceWorker(self.hand_tracker, self.gesture_predictor)

            # 3. Connect signals
            self.worker.new_frame.connect(self.update_video_feed)
            self.worker.new_prediction.connect(self.update_prediction)

            # 4. Start worker and update UI
            self.worker.start()
            self.start_button.setText("Stop Inference")
            self.set_navigation_enabled.emit(False)

        except Exception as e:
            self.prediction_label.setText(f"Error starting: {e}")
            self.start_button.setEnabled(False)

    def _stop_inference(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        if self.gesture_predictor:
            self.gesture_predictor.cleanup()

        self.worker = None
        self.gesture_predictor = None

        self.start_button.setText("Start Inference")
        self.set_navigation_enabled.emit(True)
        self.video_feed.setText("Camera Stopped")
        self.prediction_label.setText("Prediction: --")
        self.confidence_label.setText("Confidence: --")

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
        self.video_feed.setPixmap(QPixmap.fromImage(qt_image))

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
