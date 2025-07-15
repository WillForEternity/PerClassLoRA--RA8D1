import subprocess
import os
import sys
import cv2
import mediapipe as mp
import csv
import time
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRACKER_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_tracker")
TRAINING_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_training")
SIM_DIR = os.path.join(PROJECT_ROOT, "RA8D1_Simulation")

def run_command(command, cwd=PROJECT_ROOT):
    """Runs a command and yields its output line by line."""
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        shell=True, 
        cwd=cwd
    )
    for line in iter(process.stdout.readline, ''):
        yield line
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def run_setup():
    """Runs the full project setup process."""
    try:
        yield "--- Starting Project Setup ---\n"
        
        # Check for Python 3.11
        yield "[INFO] Checking for Python 3.11...\n"
        if "3.11" not in sys.version:
             yield f"[WARNING] You are running Python {sys.version}. The project uses 3.11. Proceeding, but errors may occur.\n"
        else:
            yield "[SUCCESS] Python 3.11 found.\n"

        # Create Training Venv
        yield "\n[INFO] Creating Python virtual environment for training...\n"
        yield from run_command(f'python3.11 -m venv "{TRAINING_VENV}"')
        yield "[SUCCESS] Training virtual environment created.\n"

        # Install Training Dependencies
        yield "\n[INFO] Installing training dependencies...\n"
        pip_executable = os.path.join(TRAINING_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_training.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "[SUCCESS] Training dependencies installed.\n"

        # Create Tracker Venv
        yield "\n[INFO] Creating Python virtual environment for tracking...\n"
        yield from run_command(f'python3.11 -m venv "{TRACKER_VENV}"')
        yield "[SUCCESS] Tracker virtual environment created.\n"

        # Install Tracker Dependencies
        yield "\n[INFO] Installing tracking dependencies...\n"
        pip_executable = os.path.join(TRACKER_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_tracker.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "[SUCCESS] Tracking dependencies installed.\n"

        # Build C Simulation
        yield "\n[INFO] Building C simulation...\n"
        yield from run_command('make clean && make', cwd=SIM_DIR)
        yield "[SUCCESS] C simulation built.\n"

        yield "\n--- Setup Complete! ---\n"

    except subprocess.CalledProcessError as e:
        yield f"\n[ERROR] A command failed with exit code {e.returncode}.\n"
        yield str(e)
    except Exception as e:
        yield f"\n[ERROR] An unexpected error occurred: {str(e)}\n"


# --- Hand Tracking and Data Collection ---

class HandTracker:
    """Manages camera, hand detection, and data collection logic."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.KEY_POINTS_INDICES = [0, 4, 8, 12, 16, 20]  # Wrist, Thumb, Index, Middle, Ring, Pinky tips
        self.DATA_DIR = os.path.join(PROJECT_ROOT, 'models', 'data')
        os.makedirs(self.DATA_DIR, exist_ok=True)

    def get_landmark_data(self, hand_landmarks):
        """Extracts and flattens the coordinates for the key landmarks."""
        landmark_data = []
        for i in self.KEY_POINTS_INDICES:
            lm = hand_landmarks.landmark[i]
            landmark_data.extend([lm.x, lm.y, lm.z])
        return landmark_data

    def process_frame(self, frame):
        """Processes a single video frame to find hand landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws landmarks and connections on the frame."""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def save_data(self, gesture, data):
        """Appends the collected data to the corresponding CSV file."""
        file_path = os.path.join(self.DATA_DIR, f"{gesture}.csv")
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return len(data)


# --- Gesture Prediction ---

class GesturePredictor:
    """Handles loading the ONNX model and running inference."""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.classes = ['fist', 'palm', 'pointing'] # Should match training

    def predict(self, landmark_data):
        """
        Predicts the gesture from landmark data.
        Returns the predicted gesture label and the confidence score.
        """
        # Model expects a batch, so we reshape
        input_data = np.array(landmark_data, dtype=np.float32).reshape(1, -1)
        
        # Run inference
        result = self.session.run([self.output_name], {self.input_name: input_data})[0]
        
        # Get prediction and confidence
        prediction_index = np.argmax(result)
        confidence = result[0][prediction_index]
        predicted_gesture = self.classes[prediction_index]
        
        return predicted_gesture, confidence

