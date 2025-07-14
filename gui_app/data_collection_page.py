import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from gui_app.logic import HandTracker

class CameraWorker(QThread):
    """Worker thread for handling camera input and hand tracking."""
    new_frame = pyqtSignal(np.ndarray)
    new_sample_collected = pyqtSignal(str, int)
    recording_finished = pyqtSignal()

    def __init__(self, hand_tracker):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.running = False
        self.is_recording = False
        self.current_gesture = ""

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        collected_data = []
        
        while self.running:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            results = self.hand_tracker.process_frame(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_tracker.draw_landmarks(frame, hand_landmarks)
                    if self.is_recording:
                        landmark_data = self.hand_tracker.get_landmark_data(hand_landmarks)
                        collected_data.append(landmark_data)
                        self.new_sample_collected.emit(self.current_gesture, 1)

            self.new_frame.emit(frame)

            if self.is_recording and len(collected_data) >= 100:
                self.hand_tracker.save_data(self.current_gesture, collected_data)
                collected_data = []
                self.stop_recording()
                self.recording_finished.emit()

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def start_recording(self, gesture):
        self.is_recording = True
        self.current_gesture = gesture

    def stop_recording(self):
        self.is_recording = False

class DataCollectionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTracker()
        self.worker = CameraWorker(self.hand_tracker)
        self.data_counts = {"fist": 0, "palm": 0, "pointing": 0}

        self.setup_ui()
        self.worker.new_frame.connect(self.update_video_feed)
        self.worker.new_sample_collected.connect(self.update_data_count)
        self.worker.recording_finished.connect(self.on_recording_finished)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Data Collection")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        main_layout = QHBoxLayout()
        self.video_feed = QLabel("Starting Camera...")
        self.video_feed.setFixedSize(640, 480)
        self.video_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_feed.setStyleSheet("background-color: #000; border: 1px solid #444;")
        main_layout.addWidget(self.video_feed)

        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(QLabel("1. Select Gesture:"))
        self.gesture_selector = QComboBox()
        self.gesture_selector.addItems(self.data_counts.keys())
        controls_layout.addWidget(self.gesture_selector)

        controls_layout.addWidget(QLabel("2. Start/Stop Recording:"))
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.handle_record_button)
        controls_layout.addWidget(self.record_button)

        controls_layout.addWidget(QLabel("Samples Collected:"))
        self.count_labels = {}
        for gesture in self.data_counts:
            label = QLabel(f"{gesture.capitalize()}: {self.data_counts[gesture]}")
            self.count_labels[gesture] = label
            controls_layout.addWidget(label)
        
        main_layout.addLayout(controls_layout)
        layout.addLayout(main_layout)

    def handle_record_button(self):
        if self.record_button.isChecked():
            gesture = self.gesture_selector.currentText()
            self.worker.start_recording(gesture)
            self.record_button.setText("Stop Recording")
            self.gesture_selector.setEnabled(False)
        else:
            self.worker.stop_recording()
            self.record_button.setText("Start Recording")
            self.gesture_selector.setEnabled(True)

    def on_recording_finished(self):
        self.record_button.setChecked(False)
        self.record_button.setText("Start Recording")
        self.gesture_selector.setEnabled(True)

    def update_video_feed(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.video_feed.setPixmap(QPixmap.fromImage(qt_image))

    def update_data_count(self, gesture, count):
        self.data_counts[gesture] += count
        self.count_labels[gesture].setText(f"{gesture.capitalize()}: {self.data_counts[gesture]}")

    def showEvent(self, event):
        super().showEvent(event)
        if not self.worker.isRunning():
            self.worker.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.worker.isRunning():
            self.worker.stop()
