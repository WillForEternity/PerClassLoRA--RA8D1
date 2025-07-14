import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import os
import tf2onnx
import onnx

# --- Constants ---
# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct absolute paths for data and model output
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'data')
SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model')
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.onnx')
GESTURES = ['fist', 'palm', 'pointing']
NUM_LANDMARKS = 6
NUM_COORDS = 3
INPUT_SHAPE = (NUM_LANDMARKS * NUM_COORDS,)

# --- 1. Load and Preprocess Data ---
def load_data():
    print(f"--- Loading data from: {os.path.abspath(DATA_DIR)} ---")
    data = []
    labels = []
    for i, gesture in enumerate(GESTURES):
        file_path = os.path.join(DATA_DIR, f'{gesture}.csv')
        print(f"Checking for: {file_path}")
        if not os.path.exists(file_path):
            print(f"[WARNING] Data file not found at {file_path}")
            continue
        print(f"[SUCCESS] Found and loading {gesture}.csv")
        df = pd.read_csv(file_path, header=None)
        data.append(df.values)
        labels.extend([gesture] * len(df))

    if not data:
        print("[FAILURE] No data was loaded. Please ensure CSV files exist and are not empty.")
        return None, None, None

    X = np.vstack(data)
    y = np.array(labels)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder.classes_

# --- 2. Build the Model ---
def build_model(num_classes):
    inputs = Input(shape=INPUT_SHAPE, name='input')
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting model training process...")

    # 1. Load Data
    X, y, classes = load_data()
    if X is None:
        print("Exiting due to data loading failure.")
        exit()

    print(f"Loaded {X.shape[0]} samples for {len(classes)} gestures: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Build and Train Model
    num_classes = len(classes)
    model = build_model(num_classes)
    print("\nModel Summary:")
    model.summary()

    print("\nTraining model...")
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1)

    # 3. Evaluate Model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Loss: {loss:.4f}')

    # 4. Save Model
    print(f"\nSaving model to {os.path.abspath(SAVED_MODEL_PATH)}...")
    model.export(SAVED_MODEL_PATH)
    print(f"The model is saved at: {os.path.abspath(SAVED_MODEL_PATH)}")

    # 5. Convert to ONNX
    print(f"\nConverting model to ONNX and saving to {os.path.abspath(ONNX_MODEL_PATH)}...")
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    onnx.save(onnx_model, ONNX_MODEL_PATH)
    print("Model converted to ONNX successfully.")
    print("\nModel training complete!")
