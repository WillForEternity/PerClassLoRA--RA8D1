# Technical Deep Dive: System Internals

This document provides a comprehensive, file-by-file technical explanation of the temporal hand gesture recognition system. It covers the mathematical foundations, machine learning algorithms, and software engineering principles that underpin the project.

## Core Concepts

Before dissecting each file, it's essential to understand the core concepts that govern the entire system.

1.  **Temporal Convolutional Network (TCN)**: The model is a TCN, a specialized neural network architecture designed for sequence data. Unlike RNNs, TCNs use causal, dilated convolutions to process sequences in parallel, making them computationally efficient while capturing long-range dependencies.

2.  **Hybrid Architecture (Python + C)**: The system intentionally separates high-level orchestration (Python) from performance-critical computation (C). Python's rich ecosystem is used for the GUI and data handling, while C is used to build a minimal, high-performance, and memory-efficient ML engine suitable for embedded targets.

3.  **Static Memory Allocation**: A core principle for embedded readiness. The entire C backend uses 100% static memory allocation. All required memory for the model, its gradients, and optimizer states is allocated at compile time, eliminating the risk of memory fragmentation or allocation failures on a resource-constrained device.

4.  **Consistent Data Pipeline**: The most critical concept for model accuracy. The normalization and temporal sampling methods used during inference must be mathematically identical to those used during training. Any discrepancy will lead to poor performance. This system ensures consistency through a shared normalization function and a synchronized temporal stride for both training data augmentation and inference prediction.

---

## C Backend: The Machine Learning Engine

The C backend is a self-contained, high-performance ML engine composed of two executables: `train_c` (compiled from `train_in_c.c`) for training and `ra8d1_sim` for inference.

### `RA8D1_Simulation/training_logic.h`

This header file is the data blueprint for the entire C engine. It defines the core data structures and constants.

-   **`Model` vs. `InferenceModel`**: Two key structs are defined. `Model` is a comprehensive struct containing weights, biases, gradients, and Adam optimizer states (first and second moments). This is used exclusively by `train_c`. `InferenceModel` is a lean struct containing only the weights and biases necessary for a forward pass, minimizing the memory footprint for `ra8d1_sim`.
-   **Static Allocation**: All struct members are statically-sized arrays (e.g., `float weights[SIZE]`). This is the foundation of the static memory strategy.
-   **`static_assert`**: A compile-time assertion, `static_assert(sizeof(Model) < APP_SRAM_LIMIT, ...)` ensures that the `Model` struct does not exceed the 1MB SRAM budget of the Renesas RA8D1. This is a critical safeguard for embedded development.

### `RA8D1_Simulation/training_logic.c`

This file is the mathematical core of the project, implementing the TCN and its training algorithms from scratch.

-   **`initialize_weights`**: Implements **He Initialization**. Weights are drawn from a normal distribution with a standard deviation of `sqrt(2.0 / fan_in)`. This is a standard practice for networks using ReLU-based activations, as it prevents gradients from vanishing or exploding during the initial stages of training.
-   **`forward_pass`**: Executes the TCN's forward pass. It performs a **causal convolution**, where the output at timestep `t` only depends on inputs at `t` and earlier. This is followed by a **Leaky ReLU** activation (`max(0.01*x, x)`), which prevents the "dying ReLU" problem. Finally, **Global Average Pooling** is applied across the time dimension to create a fixed-size feature vector that is fed to the output layer.
-   **`backward_pass`**: Implements backpropagation through time for the TCN. It calculates the gradients of the loss function with respect to the weights and biases of both the output layer and the TCN block.
-   **`update_weights`**: A from-scratch implementation of the **Adam Optimizer**. It updates the model's weights using the calculated gradients and the optimizer's internal state (first moment `m` and second moment `v`), including the necessary bias correction for the initial timesteps.
-   **`load_temporal_data`**: Implements **temporal data augmentation**. It reads sequences of varying lengths from CSV files and generates multiple, overlapping 20-frame windows using a `WINDOW_STRIDE` of 5. This dramatically increases the size and diversity of the training dataset.
-   **`save_model` / `load_inference_model`**: These functions perform safe, portable model serialization by writing and reading each array field-by-field. This avoids C struct padding and memory alignment issues that would otherwise make the binary model file non-portable between different systems or compiler settings.

### `RA8D1_Simulation/train_in_c.c`

This is the main entry point for the training executable. It orchestrates the entire training process.

-   **Hyperparameters**: All key training parameters (`NUM_EPOCHS`, `LEARNING_RATE`, `BETA1`, `BETA2`) are defined as constants at the top of the file.
-   **Training Loop**: The `main` function contains the primary training loop. It iterates for a fixed number of epochs, and within each epoch, it performs a training phase and a validation phase.
-   **Loss Calculation**: The `calculate_loss` function computes the **Cross-Entropy Loss**, a standard loss function for multi-class classification problems.

### `RA8D1_Simulation/main.c`

This is the inference server, designed for real-time performance and robust communication.

-   **TCP Server**: Implements a persistent, single-client TCP server that listens on port `65432`.
-   **Length-Prefix Protocol**: To handle TCP's stream-based nature, the server expects each message to be prefixed with a 4-byte unsigned integer specifying the payload length. It reads this header first, then reads exactly that many bytes to ensure it receives a complete and uncorrupted data frame.
-   **Network Byte Order**: The server correctly converts the incoming byte stream from network byte order (big-endian) to the host system's byte order using `ntohl` (network-to-host-long). This is critical for cross-platform compatibility with the Python client.

---

## Python GUI: The System Orchestrator

The Python application, built with PyQt6, manages the user workflow, handles data processing, and communicates with the C backend.

### `gui_app/logic.py`

This file contains the core business logic for the frontend, connecting the UI to the underlying data processing and communication protocols.

-   **`normalize_landmarks`**: This is the heart of the Python data pipeline and a critical component for the system's accuracy. It performs a two-step normalization process on the raw landmark data from MediaPipe:
    1.  **Translation Invariance**: It subtracts the wrist's (landmark 0) coordinates from all other landmarks, making the gesture's representation independent of its position in the camera frame.
    2.  **Scale Invariance**: It calculates the average distance of all landmarks from the new origin (the wrist) and divides all coordinates by this scale factor. This makes the gesture's representation robust to changes in hand size or distance from the camera.
-   **`HandTracker` Class**: A wrapper around the MediaPipe library. Its primary role is to process video frames, detect hand landmarks, and pass the raw landmark data to the `normalize_landmarks` function.
-   **`GesturePredictor` Class**: This class manages all communication with the C inference server.
    -   **Persistent Connection**: It establishes and maintains a persistent TCP connection to the server, with automatic reconnection logic in case of `BrokenPipeError` or `ConnectionResetError`.
    -   **Data Buffering**: It maintains a `sequence_buffer` that collects the last 20 normalized landmark frames.
    -   **Stride-Based Prediction**: This is the key to ensuring pipeline consistency. Predictions are only requested from the server every `window_stride` (5) frames. On the frames in between, it returns the last known prediction. This perfectly mirrors the temporal data augmentation used during training, ensuring the model is evaluated on data with the same temporal characteristics it was trained on.
    -   **Binary Packing**: It uses `struct.pack` with the `'!I'` and `'!f'` format specifiers to pack the message length and the float data into a binary stream with network byte order (big-endian), matching the C server's expectations.

### `gui_app/main_app.py` (and other page files)

These files define the PyQt6 user interface. They are responsible for displaying the camera feed, rendering predictions, and connecting UI button clicks (e.g., "Start Training") to the backend logic functions in `logic.py` and the C executables. They form the user-facing layer but delegate all complex processing and computation to the other components.
