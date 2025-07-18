import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QStackedWidget
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import time

from gui_app.logic import HandTracker

class CameraWorker(QThread):
    """Worker thread for handling camera input and hand tracking."""
    new_frame = pyqtSignal(np.ndarray)
    collection_update = pyqtSignal(int)
    collection_finished = pyqtSignal(str, int)

    def __init__(self, hand_tracker):
        super().__init__()
        self.hand_tracker = hand_tracker
        self._running = False
        self._collecting = False
        self.current_gesture = ""
        self.samples_to_collect = 0

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)
        collected_data = []
        last_capture_time = 0
        capture_interval = 0.0  # Capture as fast as possible

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            hand_landmarks, annotated_frame = self.hand_tracker.process_frame(frame)
            if hand_landmarks:
                # The process_frame now returns a single hand_landmarks object
                self.hand_tracker.draw_landmarks(annotated_frame, hand_landmarks)
                if self._collecting and len(collected_data) < self.samples_to_collect:
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        landmark_data = self.hand_tracker.get_landmark_data(hand_landmarks)
                        collected_data.append(landmark_data)
                        last_capture_time = current_time
                        self.collection_update.emit(len(collected_data))

            if self._collecting and len(collected_data) >= self.samples_to_collect:
                self._collecting = False
                count = self.hand_tracker.save_data(self.current_gesture, collected_data)
                self.collection_finished.emit(self.current_gesture, count)
                collected_data = []

            self.new_frame.emit(cv2.flip(annotated_frame, 1))
        cap.release()

    def start_collection(self, gesture, num_samples):
        self.current_gesture = gesture
        self.samples_to_collect = num_samples
        self._collecting = True

    def stop(self):
        self._running = False
        self._collecting = False
        self.wait()

class DataCollectionPage(QWidget):
    # Signal to control the main window's navigation
    set_navigation_enabled = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.is_setup_complete = False
        self.hand_tracker = HandTracker()
        self.worker = CameraWorker(self.hand_tracker)
        self.data_counts = {"wave": 0, "swipe_left": 0, "swipe_right": 0}

        self.setup_ui()

        self.worker.new_frame.connect(self.update_video_feed)
        self.worker.collection_update.connect(self.update_collection_progress)
        self.worker.collection_finished.connect(self.on_collection_finished)

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

        title = QLabel("Data Collection")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        content_layout = QHBoxLayout()
        self.video_feed = QLabel("Starting Camera...")
        self.video_feed.setFixedSize(640, 480)
        self.video_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_feed.setStyleSheet("background-color: #000; border: 1px solid #444;")
        content_layout.addWidget(self.video_feed)

        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(QLabel("1. Select Gesture:"))
        self.gesture_selector = QComboBox()
        self.gesture_selector.addItems(self.data_counts.keys())
        controls_layout.addWidget(self.gesture_selector)

        controls_layout.addWidget(QLabel("2. Start/Stop Recording:"))
        self.record_button = QPushButton("Start Collection")
        self.record_button.clicked.connect(self.handle_record_button)
        controls_layout.addWidget(self.record_button)

        controls_layout.addWidget(QLabel("Samples Collected:"))
        self.count_labels = {}
        for gesture in self.data_counts:
            label = QLabel(f"{gesture.capitalize()}: {self.data_counts[gesture]}")
            self.count_labels[gesture] = label
            controls_layout.addWidget(label)
        
        content_layout.addLayout(controls_layout)
        layout.addLayout(content_layout)
        self.page_stack.addWidget(main_view)

    def handle_record_button(self):
        self.set_controls_enabled(False)
        self.set_navigation_enabled.emit(False)
        selected_gesture = self.gesture_selector.currentText()
        self.record_button.setText(f"Collecting 0/100...")
        self.worker.start_collection(selected_gesture, 100)

    @pyqtSlot(int)
    def update_collection_progress(self, count):
        self.record_button.setText(f"Collecting {count}/100...")

    @pyqtSlot(str, int)
    def on_collection_finished(self, gesture, count):
        self.data_counts[gesture] += count
        self.count_labels[gesture].setText(f"{gesture.capitalize()}: {self.data_counts[gesture]}")
        self.set_controls_enabled(True)
        self.set_navigation_enabled.emit(True)

    def set_controls_enabled(self, enabled):
        self.record_button.setEnabled(enabled)
        self.gesture_selector.setEnabled(enabled)
        if enabled:
            self.record_button.setText("Start Collection")

    def update_video_feed(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.video_feed.setPixmap(QPixmap.fromImage(qt_image))

    def set_setup_status(self, is_complete):
        self.is_setup_complete = is_complete
        if self.is_setup_complete:
            self.page_stack.setCurrentIndex(1)
        else:
            self.page_stack.setCurrentIndex(0)

    def showEvent(self, event):
        super().showEvent(event)
        if self.is_setup_complete and not self.worker.isRunning():
            self.worker.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.worker.isRunning():
            self.worker.stop()
