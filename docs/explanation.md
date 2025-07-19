# Technical Deep Dive: System Internals

In this document, I'll provide a comprehensive, file-by-file technical explanation of my temporal hand gesture recognition system. I'll cover the mathematical foundations, machine learning algorithms, and software engineering principles I used to build the project.

## Core Concepts

Before dissecting each file, I should explain the core concepts that govern the system.

1.  **Temporal Convolutional Network (TCN)**: I chose a TCN for the model, a specialized neural network architecture designed for sequence data. Unlike RNNs, TCNs use causal, dilated convolutions to process sequences in parallel, which made them a computationally efficient choice for capturing long-range dependencies in the gesture data. My implementation uses a single TCN block with 8 channels and a kernel size of 3, followed by Leaky ReLU activation and Global Average Pooling.

2.  **Hybrid Architecture (Python + C)**: I designed a hybrid architecture to intentionally separate high-level orchestration (Python) from performance-critical computation (C). I used Python's rich ecosystem for the GUI and data handling, while I wrote the ML engine in C to be minimal, high-performance, and memory-efficient, making it suitable for my embedded target.

3.  **Post-Training Quantization**: To optimize the model for the Renesas RA8D1, I implemented a full post-training quantization workflow. This process converts the model's 32-bit floating-point weights into efficient 8-bit integers (INT8). This dramatically reduces the model's memory footprint and can accelerate inference, which is critical for resource-constrained devices. The entire workflow—from quantization to inference with the quantized model—is integrated into the GUI.

4.  **Static Memory Allocation**: A core principle of the C backend is static memory allocation. All memory required for the model, its gradients, and optimizer state is allocated at compile time using C `structs`. This approach eliminates dynamic memory allocation (`malloc`, `free`), preventing memory fragmentation and ensuring predictable, deterministic memory usage—a hard requirement for reliable embedded systems.

## File-by-File Breakdown

Here, I'll walk through the key files and directories, explaining their purpose and implementation.

### `start_app.sh`

This is the master script that orchestrates the entire application lifecycle. Its primary responsibilities are:
- **Environment Validation**: It checks for the existence of the C executables and the Python virtual environment, providing user-friendly error messages if they are missing.
- **Process Cleanup**: Before starting, it uses `lsof` to find and terminate any zombie processes that might be lingering on the required network port (65432), preventing "Address already in use" errors.
- **GUI Launch**: It activates the Python virtual environment and launches the main GUI application (`gui_app/main_app.py`).
- **Graceful Shutdown**: It sets up a `trap` to catch exit signals (like Ctrl+C), ensuring that any running child processes (like the C server) are properly terminated when the application closes.

### `gui_app/` - The Python GUI

This directory contains the entire PyQt6 frontend application.

#### `main_app.py`
This is the entry point for the GUI. It constructs the main window and sets up the five-page navigation system using a `QStackedWidget`. The pages are: `Setup`, `Data Collection`, `Training`, `Quantize`, and `Inference`. It connects signals between the pages to manage application state, such as enabling/disabling navigation during long-running tasks.

#### `logic.py`
This file is the heart of the Python-side logic, containing the core classes that interact with MediaPipe and the C backend.

-   **`HandTracker`**: This class manages all aspects of hand detection and data collection. It initializes a MediaPipe `Hands` object to get the 21 3D hand landmarks from the camera feed. For each frame, it normalizes the landmark data by setting the wrist as the origin (0,0,0) and scaling the coordinates based on the average distance of all landmarks from the origin. This makes the data translation- and scale-invariant. Its `save_data` method now appends new recordings to a single, consolidated CSV file for each gesture, simplifying data management.

-   **`GesturePredictor`**: This class handles real-time inference. It establishes a persistent TCP socket connection to the C inference server. Its `predict` method sends normalized landmark data (as a flat binary buffer) to the server. To match the training pipeline, it uses a stride of 5, only sending data every fifth frame. It then reads the response (prediction index and confidence), handles connection errors gracefully by attempting to reconnect, and caches the last prediction to keep the GUI display stable.

-   **`Quantizer`**: This class provides a non-blocking wrapper around the C `quantize` executable. It uses PyQt's `QProcess` to run the C program in the background, capturing its standard output in real-time and emitting signals to update the GUI. This prevents the GUI from freezing during the quantization process.

#### The Page Files (`training_page.py`, `inference_page.py`, etc.)
Each page script manages a specific part of the workflow. Key implementations include:
-   `training_page.py`: Launches the C `train_c` executable. It now dynamically loads the user-defined gesture list from `gestures.json` and passes the names as command-line arguments to the C program.
-   `quantize_page.py`: Uses the `Quantizer` class from `logic.py` to run the quantization process and display the output.
-   `inference_page.py`: Manages the lifecycle of the C inference server (`ra8d1_sim`). It starts the server when the page is loaded and provides a dropdown menu for the user to select between the float (`c_model.bin`) and quantized (`c_model_quantized.bin`) models. It passes the chosen model path to the server as a command-line argument.

### `RA8D1_Simulation/` - The C Backend

This directory contains the high-performance C code for all machine learning tasks.

#### `training_logic.h`
This header file is the single source of truth for the system's data structures. It defines:
-   **`Model`**: The main struct for training. It contains arrays for weights, biases, gradients, and the Adam optimizer's internal state (`m` and `v` moments) for both the TCN and output layers.
-   **`InferenceModel`**: A lightweight version of the `Model` struct, containing only the weights and biases needed for a forward pass. This is what the inference server uses to minimize its memory footprint.
-   **`QuantizedModel`**: A struct containing `int8_t` arrays for the quantized weights and biases.
-   A `static_assert` that checks the `sizeof(Model)` against the `APP_SRAM_LIMIT` at compile time, providing an immediate failure if the model grows too large for the target MCU.

#### `training_logic.c`
This file contains the core implementation of the TCN model.
-   `load_temporal_data`: Implements a two-pass sliding window algorithm to read the gesture data. In the first pass, it counts the total number of valid sequences across all gesture files. In the second pass, it allocates memory and populates the data and label arrays. It now reads from a single, consolidated CSV file per gesture.
-   `forward_pass`: Executes the TCN forward pass: a causal, dilated convolution, followed by a Leaky ReLU activation and Global Average Pooling across the time dimension.
-   `backward_pass` & `update_weights`: Implement the backpropagation-through-time algorithm and the Adam optimizer to update the model's weights based on the calculated gradients.

#### `train_in_c.c`
This is the `main` function for the training executable. It now parses command-line arguments to get the list of gestures to train on. It calls `load_temporal_data` to load the data, then iterates through the training epochs, calling the `forward_pass`, `backward_pass`, and `update_weights` functions. Finally, it saves the trained weights to `c_model.bin`.

#### `quantize.c`
This is the `main` function for the quantization executable. It loads the float model (`c_model.bin`), converts its weights to `int8_t` using simple symmetric quantization (`value * 127`), and saves the result to `c_model_quantized.bin`. It also prints the size of the original and quantized models to show the memory savings.

#### `main.c`
This is the `main` function for the inference server. It accepts a command-line argument specifying which model file to use. It dynamically detects whether to load a float or quantized model based on the filename. It then enters a loop, listening for incoming TCP connections, reading the landmark data, running the appropriate forward pass (`forward_pass_inference` or `forward_pass_quantized`), and sending the prediction back to the Python client.

3.  **Static Memory Allocation**: For embedded readiness, I used 100% static memory allocation in the C backend. I allocate all required memory for the model, its gradients, and optimizer states at compile time. This eliminates the risk of memory fragmentation or allocation failures on a resource-constrained device.

4.  **Consistent Data Pipeline**: This was the most critical concept for achieving high accuracy. The normalization and temporal sampling methods I used during inference had to be mathematically identical to those used during training. To prevent discrepancies, I implemented a shared normalization function and a synchronized temporal stride for both training data augmentation and real-time inference.

---

## C Backend: The Machine Learning Engine

I built the C backend as a self-contained, high-performance ML engine. It's composed of two executables: `train_c` for training and `ra8d1_sim` for inference.

### `RA8D1_Simulation/training_logic.h`

This header file is the data blueprint for my entire C engine. Here, I defined the core data structures and constants.

-   **`Model` vs. `InferenceModel`**: I defined two key structs. `Model` is a comprehensive struct containing everything needed for training: weights, biases, gradients, and Adam optimizer states. `InferenceModel` is a lean version with only the weights and biases necessary for a forward pass, which minimizes the memory footprint for the inference engine.
-   **Static Allocation**: All struct members are statically-sized arrays (e.g., `float weights[SIZE]`). This was the foundation of my static memory strategy.
-   **`static_assert`**: I added a compile-time assertion, `static_assert(sizeof(Model) < APP_SRAM_LIMIT, ...)` to ensure the `Model` struct never exceeds the 1MB SRAM budget of the Renesas RA8D1. This was a critical safeguard for my embedded development workflow.

### `RA8D1_Simulation/training_logic.c`

This file is the mathematical core of the project, where I implemented the TCN and its training algorithms from scratch.

-   **`initialize_weights`**: I implemented **He Initialization**, drawing weights from a normal distribution with a standard deviation of `sqrt(2.0 / fan_in)`. This is a standard practice for networks using ReLU, and it helped prevent gradients from vanishing or exploding during training.
-   **`forward_pass`**: Here, I execute the TCN's forward pass. It performs a **causal convolution**, followed by a **Leaky ReLU** activation (`max(0.01*x, x)`) to prevent the "dying ReLU" problem. Finally, I apply **Global Average Pooling** across the time dimension to create a fixed-size feature vector for the output layer.
-   **`backward_pass`**: My implementation of backpropagation through time for the TCN. It calculates the gradients of the loss function with respect to the weights and biases of the entire network.
-   **`update_weights`**: My from-scratch implementation of the **Adam Optimizer**. It updates the model's weights using the calculated gradients and the optimizer's internal state, including the necessary bias correction.
-   **`load_temporal_data`**: Here, I implemented **temporal data augmentation**. My function reads sequences from CSV files and generates multiple, overlapping 20-frame windows using a `WINDOW_STRIDE` of 5. This dramatically increased the size and diversity of my training dataset.
-   **`save_model` / `load_inference_model`**: I wrote these functions to perform safe, portable model serialization by writing and reading each array field-by-field. This approach avoids C struct padding and alignment issues that would otherwise make the binary model file non-portable.

### `RA8D1_Simulation/train_in_c.c`

This is the main entry point for my training executable, which orchestrates the entire training process.

-   **Hyperparameters**: I defined all key training parameters (`NUM_EPOCHS`, `LEARNING_RATE`, etc.) as constants at the top of the file for easy tuning.
-   **Training Loop**: In the `main` function, I created the primary training loop that iterates for a fixed number of epochs, running a training and validation phase in each.
-   **Loss Calculation**: I wrote the `calculate_loss` function to compute the **Cross-Entropy Loss**, which is standard for multi-class classification problems.

### `RA8D1_Simulation/main.c`

This is my inference server, which I designed for real-time performance and robust communication.

-   **TCP Server**: I implemented a persistent, single-client TCP server that listens on port `65432`.
-   **Length-Prefix Protocol**: To handle TCP's stream-based nature, I designed a simple protocol where every message is prefixed with a 4-byte unsigned integer specifying the payload length. The server reads this header first to ensure it receives a complete data frame.
-   **Network Byte Order**: I made sure the server correctly converts the incoming byte stream from network byte order to the host system's byte order using `ntohl`. This was critical for cross-platform compatibility with the Python client.
-   **Model Loading and Lifecycle**: The server attempts to load `c_model.bin` only once at startup. Because it doesn't automatically reload, I made the Python GUI responsible for restarting this server process after training to force it to load the new model file.

---

## Python GUI: The System Orchestrator

I built the Python application with PyQt6 to manage the user workflow, handle data processing, and communicate with the C backend.

### `gui_app/logic.py`

This file contains the core business logic, connecting the UI to the underlying data processing and communication protocols.

-   **`normalize_landmarks`**: This function is the heart of my Python data pipeline. I implemented a two-step normalization on the raw landmark data from MediaPipe:
    1.  **Translation Invariance**: I subtract the wrist's coordinates from all other landmarks, making the gesture independent of its position in the camera frame.
    2.  **Scale Invariance**: I calculate the average distance of all landmarks from the new origin (the wrist) and divide all coordinates by this scale factor. This makes the gesture robust to changes in hand size or distance from the camera.
-   **`HandTracker` Class**: I created this as a wrapper around the MediaPipe library to process video frames, detect hand landmarks, and pass the raw data to my `normalize_landmarks` function.
-   **`GesturePredictor` Class**: This class manages all communication with the C inference server.
    -   **Persistent Connection**: I implemented logic to establish and maintain a persistent TCP connection, with automatic reconnection in case of errors.
    -   **Data Buffering**: It maintains a `sequence_buffer` that collects the last 20 normalized landmark frames.
    -   **Stride-Based Prediction**: This was my key to ensuring pipeline consistency. I only request predictions from the server every 5 frames (`window_stride`). On frames in between, it returns the last known prediction. This perfectly mirrors the data augmentation I used during training.
    -   **Binary Packing**: I used `struct.pack` with network byte order (`'!'`) to pack data into a binary stream, matching the C server's expectations.

### `gui_app/main_app.py` (and other page files)

These files define the PyQt6 UI. They display the camera feed, render predictions, and connect UI buttons to my backend logic. The `training_page.py` module has the key responsibility of restarting the C inference server after a training run to ensure the new model is loaded.
