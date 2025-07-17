import subprocess
import os
import sys
import cv2
import mediapipe as mp
import csv
import time
import numpy as np
import socket

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

        self.DATA_DIR = os.path.join(PROJECT_ROOT, 'models', 'data')
        os.makedirs(self.DATA_DIR, exist_ok=True)

    def _normalize_landmarks(self, landmarks_np):
        """Normalizes a numpy array of landmarks.
        Helper function to be used by both inference and data collection.
        """
        # 1. Normalize by subtracting the wrist (landmark 0) position
        origin = landmarks_np[0].copy()
        relative_landmarks = landmarks_np - origin

        # 2. Calculate scale factor (average distance from origin)
        distances = np.linalg.norm(relative_landmarks, axis=1)
        scale_factor = np.mean(distances)
        if scale_factor < 1e-6: # Avoid division by zero
            scale_factor = 1

        # 3. Scale the data
        normalized_landmarks = relative_landmarks / scale_factor

        # 4. Exclude the wrist landmark (it's the origin) and flatten
        return normalized_landmarks[1:].flatten().tolist()

    def get_landmark_data(self, hand_landmarks):
        """Extracts, normalizes, and flattens all 21 landmark coordinates for inference."""
        if not hand_landmarks:
            return None
        
        # Extract all 21 landmarks into a numpy array
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        return self._normalize_landmarks(landmarks_np)

    def process_frame(self, frame):
        """Processes a single video frame, finds, and draws hand landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_landmarks = None
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            # Draw the landmarks on the original frame
            self.draw_landmarks(frame, hand_landmarks)

        return hand_landmarks, frame # Return landmarks and the (possibly annotated) frame

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws landmarks and connections on the frame."""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def save_data(self, gesture, data):
        """Saves a gesture sequence to a new CSV file."""
        gesture_dir = os.path.join(self.DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Find the next available file number
        existing_files = os.listdir(gesture_dir)
        next_file_num = 1
        while f"{gesture}_{next_file_num}.csv" in existing_files:
            next_file_num += 1

        file_path = os.path.join(gesture_dir, f"{gesture}_{next_file_num}.csv")
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for frame_landmarks_flat in data:
                if not frame_landmarks_flat: continue # Skip empty frames

                # Reshape the flat list into a (21, 3) numpy array
                landmarks_np = np.array(frame_landmarks_flat).reshape(-1, 3)
                # Use the centralized normalization method
                row = self._normalize_landmarks(landmarks_np)

                writer.writerow(row)
        
        # Return 1 to indicate one sequence was saved
        return 1


# --- Gesture Prediction ---

class GesturePredictor:
    """
    Manages communication with the C-based inference server to get temporal
    gesture predictions from a sequence of landmark data using a persistent connection.
    """
    def __init__(self):
        self.host = 'localhost'
        self.port = 65432
        self.client_socket = None
        self.rfile = None # For buffered reading
        self.classes = ['wave', 'swipe_left', 'swipe_right', 'Gesture Not Recognized']
        self.confidence_threshold = 0.70 # 70% confidence required
        self.sequence_length = 100 # Must match SEQUENCE_LENGTH in C backend
        self.num_features = 21 * 3
        self.sequence_buffer = [[0.0] * self.num_features for _ in range(self.sequence_length)]
        self._connect() # Establish initial connection

    def _connect(self):
        """Establishes or re-establishes the persistent connection to the C server."""
        self.cleanup(is_reconnecting=True)
        try:
            print("[GesturePredictor] Attempting to connect to C inference server...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.rfile = self.client_socket.makefile('r') # Create buffered reader
            print("[GesturePredictor] Connection to C server successful.")
        except ConnectionRefusedError:
            print("[GesturePredictor] Connection refused. Is the C server running?")
            self.client_socket = None
            self.rfile = None

    def predict(self, landmark_data):
        """
        Buffers landmark data, sends it over the persistent connection for inference,
        and handles reconnections if necessary.
        """
        if landmark_data is None:
            return "No Hand Present", 0.0

        self.sequence_buffer.append(landmark_data)
        self.sequence_buffer.pop(0)

        if not self.client_socket or not self.rfile:
            self._connect()
            if not self.client_socket:
                return "Connecting...", 0.0

        try:
            # 1. Format data
            flat_sequence = [item for sublist in self.sequence_buffer for item in sublist]
            data_string = ",".join(map(str, flat_sequence)) + "\n"

            # 2. Send data and get response over the persistent connection
            self.client_socket.sendall(data_string.encode('utf-8'))
            response = self.rfile.readline().strip() # Read a single, complete line

            # 3. Parse and return the prediction
            if not response:
                # This can happen if the server closes the connection cleanly
                print("[GesturePredictor] Empty response from server. Reconnecting...")
                self._connect()
                return "Connecting...", 0.0
            
            parts = response.split(',')
            prediction_index = int(parts[0])
            confidence = float(parts[1])

            if confidence < self.confidence_threshold:
                return self.classes[-1], confidence
            else:
                return self.classes[prediction_index], confidence

        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[GesturePredictor] Connection lost: {e}. Reconnecting...")
            self._connect()
            return "Connecting...", 0.0
        except Exception as e:
            print(f"[GesturePredictor] An error occurred during prediction: {e}")
            # Don't reconnect on general errors, could be a data issue
            return "Error", 0.0

    def cleanup(self, is_reconnecting=False):
        """Cleans up resources for the GesturePredictor, closing the socket."""
        if not is_reconnecting:
            print("Cleaning up GesturePredictor...")
        
        if self.rfile:
            try: self.rfile.close()
            except Exception as e: print(f"Error closing reader file: {e}")
        
        if self.client_socket:
            try: self.client_socket.close()
            except Exception as e: print(f"Error closing client socket: {e}")

        self.rfile = None
        self.client_socket = None
        
        if not is_reconnecting:
            self.sequence_buffer.clear()
            print("GesturePredictor cleanup complete.")

