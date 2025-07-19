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
from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QTimer

from gui_app.config import load_gestures

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TRACKER_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_tracker")
TRAINING_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_training")
SIM_DIR = os.path.join(PROJECT_ROOT, "RA8D1_Simulation")

def run_command(command, cwd=PROJECT_ROOT):
    """Run a command and yield its output."""
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
    """Run the full project setup."""
    try:
        yield "Starting project setup...\n"
        
        # Check for Python 3.11
        yield "Checking for Python 3.11...\n"
        if "3.11" not in sys.version:
             yield f"Warning: Running Python {sys.version}, not 3.11. Errors may occur.\n"
        else:
            yield "Python 3.11 found.\n"

        # Create Training Venv
        yield "\nCreating training venv...\n"
        yield from run_command(f'python3.11 -m venv "{TRAINING_VENV}"')
        yield "Training venv created.\n"

        # Install Training Dependencies
        yield "\nInstalling training dependencies...\n"
        pip_executable = os.path.join(TRAINING_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_training.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "Training dependencies installed.\n"

        # Create Tracker Venv
        yield "\nCreating tracking venv...\n"
        yield from run_command(f'python3.11 -m venv "{TRACKER_VENV}"')
        yield "Tracking venv created.\n"

        # Install Tracker Dependencies
        yield "\nInstalling tracking dependencies...\n"
        pip_executable = os.path.join(TRACKER_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_tracker.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "Tracking dependencies installed.\n"

        # Build C Simulation
        yield "\nBuilding C simulation...\n"
        yield from run_command('make clean && make', cwd=SIM_DIR)
        yield "C simulation built.\n"

        yield "\nSetup complete!\n"

    except subprocess.CalledProcessError as e:
        yield f"\nError: Command failed (exit code {e.returncode}).\n"
        yield str(e)
    except Exception as e:
        yield f"\nError: An unexpected error occurred: {e}\n"


# Data Normalization

def normalize_landmarks(landmarks_np):
    """Normalize landmarks for inference and data collection."""
    # Set wrist as origin
    origin = landmarks_np[0].copy()
    relative_landmarks = landmarks_np - origin

    # Calculate scale factor (avg distance from origin)
    distances = np.linalg.norm(relative_landmarks, axis=1)
    scale_factor = np.mean(distances)
    if scale_factor < 1e-6: # Avoid division by zero
        scale_factor = 1

    # Scale data
    normalized_landmarks = relative_landmarks / scale_factor

    # Flatten and return all 21 landmarks
    return normalized_landmarks.flatten().tolist()

# Hand Tracking and Data Collection

class HandTracker:
    """Manage camera, hand detection, and data collection."""

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
        """Extract and normalize hand landmarks for inference."""
        if not hand_landmarks:
            return None
        
        # Extract all 21 landmarks as numpy array
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Normalize and exclude wrist (returns 60 floats)
        return normalize_landmarks(landmarks_np)

    def process_frame(self, frame):
        """Process a video frame to find and draw hand landmarks."""
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
        """Draw landmarks and connections on the frame."""
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def save_data(self, gesture, data):
        """Save a gesture sequence to a single CSV file."""
        data_dir = os.path.join(MODELS_DIR, 'data', gesture)
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f'{gesture}.csv')

        # Check if the file exists to decide whether to write the header
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header only if the file is new
            if not file_exists:
                header = []
                for i in range(1, 21): # 20 landmarks
                    header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
                writer.writerow(header)
            
            # Append new data
            for frame_data in data:
                writer.writerow(frame_data)
        
        print(f"Appended {len(data)} samples to {file_path}")
        return len(data)


# Gesture Prediction

class GesturePredictor:
    """Get temporal gesture predictions from the C inference server."""
    def __init__(self):
        self.host = 'localhost'
        self.port = 65432
        self.client_socket = None
        self.rfile = None # For buffered reading
        self.last_confidence = 0.0
        self.confidence_threshold = 0.5
        self.classes = load_gestures() + ["No Hand Present"] # Load custom gestures
        self.connection_timer = QTimer()
        self.sequence_length = 20 # Must match SEQUENCE_LENGTH in C backend
        self.window_stride = 5 # Must match WINDOW_STRIDE in C training code
        self.num_features = 60 # We receive 60 features per frame (20 landmarks × 3 coords)
        self.sequence_buffer = [] # Buffer will store normalized landmarks
        self.frame_counter = 0  # Track frames for stride-based prediction
        self.last_prediction = "Collecting data..."
        self.last_confidence = 0.0
        self._connect() # Establish initial connection

    def _connect(self):
        """Connect (or reconnect) to the C server."""
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
        # landmark_data is already normalized and contains 63 floats (21 landmarks × 3 coords)
        self.sequence_buffer.append(landmark_data)
        self.frame_counter += 1
        
        # If buffer is full, remove the oldest frame
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        # Ensure full sequence at a stride boundary
        if len(self.sequence_buffer) < self.sequence_length:
            return "Collecting data...", 0.0
        
        # Predict every WINDOW_STRIDE frames to match training
        if (self.frame_counter - self.sequence_length) % self.window_stride != 0:
            # Return last prediction to keep UI stable
            return self.last_prediction, self.last_confidence

        if not self.client_socket or not self.rfile:
            self._connect()
            if not self.client_socket:
                return "Connecting...", 0.0

        try:
            # Flatten sequence buffer (20 frames * 60 floats = 1200 floats)
            normalized_sequence = []
            for frame_landmarks in self.sequence_buffer:
                normalized_sequence.extend(frame_landmarks)

            # Verify data size
            expected_size = self.sequence_length * 63  # 20 frames × 63 floats per frame
            if len(normalized_sequence) != expected_size:
                print(f"[GesturePredictor] Data size mismatch: got {len(normalized_sequence)}, expected {expected_size}")
                return "Data Error", 0.0

            # Pack data as binary stream of floats (network byte order)
            format_string = '!' + 'f' * len(normalized_sequence)
            data_bytes = struct.pack(format_string, *normalized_sequence)

            # Prepend message length and send
            msg_len = len(data_bytes)
            len_prefix = struct.pack('!I', msg_len) # Pack as 4-byte unsigned int, network byte order
            self.client_socket.sendall(len_prefix + data_bytes)

            # Read response
            response = self.client_socket.recv(1024).decode('utf-8').strip()

            # Parse and return prediction
            if not response:
                # Server closed connection
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
            # Don't reconnect on general errors (could be data issue)
            return "Error", 0.0

    def cleanup(self, is_reconnecting=False):
        """Close the socket and clean up resources."""
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

class Quantizer(QObject):
    """Manages the C model quantization process."""
    output_received = pyqtSignal(str)
    quantization_finished = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.process = None

    def run_quantization(self):
        """Runs the C quantization executable as a separate process."""
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.output_received.emit("Quantization is already in progress.")
            return

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_ready_read)
        self.process.finished.connect(self.on_process_finished)

        # Define paths
        quantize_executable = os.path.join(SIM_DIR, "quantize")
        input_model_path = os.path.join(MODELS_DIR, "c_model.bin")
        output_model_path = os.path.join(SIM_DIR, "c_model_quantized.bin")

        # Check for executable
        if not os.path.exists(quantize_executable):
            self.output_received.emit(f"Error: Quantize executable not found at {quantize_executable}. Please compile the C code first.")
            self.quantization_finished.emit(-1)
            return
        
        # Check for input model
        if not os.path.exists(input_model_path):
            self.output_received.emit(f"Error: Base model not found at {input_model_path}. Please train a model first.")
            self.quantization_finished.emit(-1)
            return

        # Run the quantization process
        self.output_received.emit("Starting quantization process...\n")
        self.process.start(quantize_executable, [input_model_path, output_model_path])

    def on_ready_read(self):
        """Emits the output from the C executable."""
        data = self.process.readAllStandardOutput().data().decode().strip()
        if data:
            self.output_received.emit(data)

    def on_process_finished(self, exit_code, exit_status):
        """Handles the completion of the quantization process."""
        if exit_status == QProcess.ExitStatus.CrashExit:
            self.output_received.emit("\nError: The quantization process crashed.")
        self.quantization_finished.emit(exit_code)
        self.process = None


