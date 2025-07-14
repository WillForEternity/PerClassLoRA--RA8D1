import cv2
import mediapipe as mp
import socket
import time
import argparse
import csv
import os

# --- Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Constants ---
HOST = '127.0.0.1'
PORT = 65432
KEY_POINTS_INDICES = [0, 4, 8, 12, 16, 20] # Wrist, Thumb, Index, Middle, Ring, Pinky tips
DATA_DIR = '../models/data'

# --- Functions ---

def get_landmark_data(hand_landmarks):
    """Extracts and flattens the coordinates for the key landmarks."""
    landmark_data = []
    for i in KEY_POINTS_INDICES:
        lm = hand_landmarks.landmark[i]
        landmark_data.extend([lm.x, lm.y, lm.z])
    return landmark_data

def run_inference_mode():
    """Connects to the C simulation server and streams hand landmark data."""
    print("Starting INFERENCE mode...")
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    retries = 0
    while True:
        try:
            client_socket.connect((HOST, PORT))
            print(f"Successfully connected to C server at {HOST}:{PORT}")
            break
        except ConnectionRefusedError:
            retries += 1
            if retries >= 10:
                print("Connection failed after multiple retries. Is the C simulation running?")
                return
            print(f"Connection refused. Retrying in 2 seconds... ({retries}/10)")
            time.sleep(2)

    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_data = get_landmark_data(hand_landmarks)
                    data_string = ",".join(map(str, landmark_data))
                    try:
                        client_socket.sendall(data_string.encode('utf-8'))
                    except (BrokenPipeError, ConnectionResetError):
                        print("Server disconnected.")
                        return

            cv2.imshow('Inference Mode', image)
            if cv2.waitKey(5) & 0xFF == 27: break
            time.sleep(0.05) # Add a small delay to prevent overwhelming the C client
    finally:
        print("Closing connection and camera.")
        cap.release()
        client_socket.close()
        cv2.destroyAllWindows()

def run_collection_mode():
    """Collects training data for specified hand gestures."""
    print("Starting DATA COLLECTION mode...")
    os.makedirs(DATA_DIR, exist_ok=True)
    gestures = ["fist", "palm", "pointing"]
    samples_per_gesture = 100

    cap = cv2.VideoCapture(0)

    for gesture in gestures:
        print(f"\nPrepare to show the '{gesture.upper()}' gesture.")
        print("Starting in 5 seconds...")
        time.sleep(5)
        print("RECORDING...")

        collected_data = []
        while len(collected_data) < samples_per_gesture:
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_data = get_landmark_data(hand_landmarks)
                    collected_data.append(landmark_data)
                    # Display progress
                    cv2.putText(image, f'Recording {gesture}: {len(collected_data)}/{samples_per_gesture}', 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Data Collection', image)
            if cv2.waitKey(5) & 0xFF == 27: 
                cap.release()
                return

        # Save the collected data to a CSV file
        file_path = os.path.join(DATA_DIR, f"{gesture}.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(collected_data)
        print(f"Saved {len(collected_data)} samples to {file_path}")

    cap.release()
    print("\nData collection complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand Tracking Application')
    parser.add_argument('--mode', type=str, default='inference', choices=['inference', 'collect'],
                        help='The mode to run the application in: `inference` or `collect`.')
    args = parser.parse_args()

    if args.mode == 'collect':
        run_collection_mode()
    else:
        run_inference_mode()
