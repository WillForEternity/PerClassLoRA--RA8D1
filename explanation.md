# Project Explanation: From Simulation to GUI

This document outlines the evolution of the Hand Gesture Recognition project, from its initial concept as a command-line-based simulation to its final implementation as a full-featured graphical user interface (GUI) application.

## Initial Goal: C-Based MCU Simulation

The project began with the ambitious goal of simulating a complete, adaptive machine learning workflow on a resource-constrained microcontroller (the Renesas RA8D1). The plan was to:

1.  **Develop in C:** Write all core logic in C to mirror the embedded environment.
2.  **Simulate Data Input:** Use a Python script with `MediaPipe` to stream hand landmark data over a TCP socket to the C application.
3.  **Train Offline:** Train a model in Python and export it to ONNX.
4.  **Run Inference in C:** Use a C-based ONNX runtime to perform inference on the live data stream.
5.  **Simulate On-Device Learning:** Implement a lightweight adaptation technique (like Per-Class LoRA) in C.

This approach, while technically rigorous, presented significant challenges in usability and development speed. The workflow was fragmented across multiple terminals and required manual coordination between the Python data source and the C simulation.

## Evolution: The Shift to a Python GUI

To create a more integrated and user-friendly experience, the project pivoted from a C-based simulation to a Python-native GUI application. This shift prioritized ease of use and rapid development while still demonstrating the core machine learning workflow.

The final application is a self-contained tool that guides the user through every step of the process.

### Architectural Design

The final architecture is built around a central PyQt6 GUI application that manages the entire workflow. Key design decisions include:

- **Modular, Page-Based UI:** The application is divided into four pages (`Setup`, `Data Collection`, `Training`, `Inference`), each handling a specific part of the workflow. This makes the application easy to navigate and understand.

- **Isolated Python Environments:** To avoid complex dependency conflicts (especially with TensorFlow on Apple Silicon), the application uses two separate virtual environments:
    - `venv_gui`: A lightweight environment for the PyQt6 GUI and its dependencies.
    - `venv_training`: A dedicated environment with specific versions of TensorFlow, tf2onnx, and other libraries required for model training.

- **Asynchronous Operations with QThread:** Long-running tasks like data collection, training, and inference are offloaded to separate `QThread` workers. This ensures the GUI remains responsive and never freezes, providing a smooth user experience.

- **Live Feedback Mechanisms:**
    - **Training:** The training script uses a `CSVLogger` callback to write metrics to a file after each epoch. The GUI then reads this file on a timer to update a Matplotlib graph in real-time.
    - **Inference:** The inference worker captures camera frames, runs them through the hand tracker and ONNX model, and emits signals to the main thread to update the video feed and prediction labels.

### Core Components

- **`main_app.py`:** The entry point of the application. It sets up the main window, manages the navigation between pages, and connects signals and slots.

- **`logic.py`:** Contains the core, non-UI logic:
    - `HandTracker`: A wrapper around `MediaPipe` to process camera frames and extract hand landmarks.
    - `GesturePredictor`: A class that loads the trained `model.onnx` file and uses `onnxruntime` to run predictions.

- **Page Modules (`*_page.py`):** Each page is a self-contained module with its own UI layout and worker thread for handling its specific task. This modular design makes the codebase clean and easy to maintain.

- **`train_model.py`:** The script responsible for loading the collected CSV data, building and training a TensorFlow model, and exporting it to the ONNX format for inference.

## Conclusion

By transitioning from a complex, multi-language simulation to a unified Python GUI, the project successfully delivered a robust and user-friendly tool for hand gesture recognition. It provides a complete, end-to-end workflow that is accessible to a broader audience while still demonstrating the key principles of a practical machine learning pipeline.

