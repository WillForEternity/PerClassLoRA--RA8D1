import subprocess
import os
import sys
import cv2
import mediapipe as mp
import csv
import time
import numpy as np
import socket
import struct

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


# --- Data Normalization ---

def normalize_landmarks(landmarks_np):
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



    def get_landmark_data(self, hand_landmarks):
        """Extracts and normalizes hand landmarks for inference.
        Returns normalized data ready for the C server (60 floats).
        """
        if not hand_landmarks:
            return None
        
        # Extract all 21 landmarks as numpy array
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Normalize and exclude wrist (returns 60 floats)
        return normalize_landmarks(landmarks_np)

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

                # Data is already normalized from get_landmark_data() - just save it directly
                # frame_landmarks_flat is already a list of 60 floats (20 landmarks × 3 coords, wrist excluded)
                writer.writerow(frame_landmarks_flat)
        
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
        self.sequence_length = 20 # Must match SEQUENCE_LENGTH in C backend
        self.window_stride = 5 # Must match WINDOW_STRIDE in C training code
        self.num_features = 60 # We receive 60 features per frame (20 landmarks × 3 coords)
        self.sequence_buffer = [] # Buffer will store normalized landmarks
        self.frame_counter = 0  # Track frames for stride-based prediction
        self.last_prediction = "Collecting data..."
        self.last_confidence = 0.0
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
        Uses stride-based prediction to match training temporal sampling.
        """
        # If no hand is detected, clear the buffer and reset frame counter
        if landmark_data is None:
            self.sequence_buffer.clear()
            self.frame_counter = 0
            self.last_prediction = "No Hand Present"
            self.last_confidence = 0.0
            return self.last_prediction, self.last_confidence

        # A hand is present, so add the new data to our buffer.
        # landmark_data is already normalized and contains 60 floats (20 landmarks × 3 coords)
        self.sequence_buffer.append(landmark_data)
        self.frame_counter += 1
        
        # If buffer is full, remove the oldest frame
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        # Only proceed if we have a full sequence AND we're at a stride boundary
        if len(self.sequence_buffer) < self.sequence_length:
            return "Collecting data...", 0.0
        
        # Only make predictions every WINDOW_STRIDE frames to match training pattern
        if (self.frame_counter - self.sequence_length) % self.window_stride != 0:
            # Skip inference, return last known prediction to keep UI stable
            return self.last_prediction, self.last_confidence

        if not self.client_socket or not self.rfile:
            self._connect()
            if not self.client_socket:
                return "Connecting...", 0.0

        try:
            # 1. Flatten the sequence buffer - data is already normalized
            # Each frame has 60 floats, so total should be 20 frames × 60 floats = 1200 floats
            normalized_sequence = []
            for frame_landmarks in self.sequence_buffer:
                normalized_sequence.extend(frame_landmarks)

            # Verify we have the expected amount of data
            expected_size = self.sequence_length * 60  # 20 frames × 60 floats per frame
            if len(normalized_sequence) != expected_size:
                print(f"[GesturePredictor] Data size mismatch: got {len(normalized_sequence)}, expected {expected_size}")
                return "Data Error", 0.0

            # 2. Format data for sending as a raw binary stream of floats
            # The format string consists of a float 'f' for each value in the sequence.
            # '!' ensures network byte order (big-endian).
            format_string = '!' + 'f' * len(normalized_sequence)
            data_bytes = struct.pack(format_string, *normalized_sequence)

            # 3. Pack message with length prefix and send
            msg_len = len(data_bytes)
            len_prefix = struct.pack('!I', msg_len) # Pack as 4-byte unsigned int, network byte order
            self.client_socket.sendall(len_prefix + data_bytes)

            # 4. Read response
            response = self.client_socket.recv(1024).decode('utf-8').strip()

            # 5. Parse and return the prediction
            if not response:
                # This can happen if the server closes the connection cleanly
                print("[GesturePredictor] Empty response from server. Reconnecting...")
                self._connect()
                return "Connecting...", 0.0
            
            parts = response.split(',')
            prediction_index = int(parts[0])
            confidence = float(parts[1])

            if confidence < self.confidence_threshold:
                self.last_prediction = self.classes[-1]
                self.last_confidence = confidence
            else:
                self.last_prediction = self.classes[prediction_index]
                self.last_confidence = confidence
            return self.last_prediction, self.last_confidence

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

