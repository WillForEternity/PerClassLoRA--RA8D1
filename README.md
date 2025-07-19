# Temporal Hand Gesture Recognition for Renesas RA8D1

[![Built For:](https://img.shields.io/badge/Built%20For-Renesas%20RA8D1-blue)](https://www.renesas.com/us/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ra8d1-480-mhz-arm-cortex-m85-based-microcontroller-helium-and-trustzone)
[![Memory](https://img.shields.io/badge/Memory-%3C1MB%20SRAM-orange)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-98%25%2B-success)]()
[![Quantization](https://img.shields.io/badge/Quantization-INT8%20Ready-blueviolet)]()

## 1. Project Overview

This project delivers a real-time temporal hand gesture recognition system optimized for the Renesas RA8D1 microcontroller. It features a custom-built Temporal Convolutional Network (TCN) designed to operate within a strict 1MB SRAM memory constraint, demonstrating the successful deployment of advanced machine learning on a resource-constrained embedded platform.

The system is now fully functional and robust. After a rigorous debugging process that resolved critical bugs in the C-based Adam optimizer and the server's model-loading lifecycle, the entire pipeline—from data collection to training and real-time inference—operates with high accuracy and stability.

## 2. Getting Started

### Prerequisites
- A C compiler (e.g., `gcc` or `clang`)
- `make`
- **Python 3.11** (This specific version is required for GUI dependencies on macOS).

### Automated Execution

The entire application is managed by a single master script. This command builds the C code, sets up the Python environment, starts the C server, and launches the GUI in the correct order.

```bash
# Clone the repository and run the application
git clone https://github.com/WillForEternity/PerClassLoRA--RA8D1.git
cd PerClassLoRA--RA8D1
./start_app.sh
```

The script handles all dependencies and ensures the C server is running before launching the GUI. It also gracefully shuts down all processes on exit.

### Manual Execution (Advanced)
For developers who prefer manual control, the C backend can be compiled and run using the provided `Makefile`.

```bash
# Navigate to the C backend directory
cd RA8D1_Simulation

# Build both the training and inference executables
make all

# To run the executables (after building)
./train_c        # Run the training process
./ra8d1_sim      # Run the inference server

# To clean all build artifacts
make clean
```

## 3. Features

- **Quantization Support:** Full workflow for quantizing the trained model, including a C-based quantization tool and GUI integration for easy export and inference with quantized models. The quantization process now also displays the final compressed model size, providing immediate feedback on efficiency gains.
- **Customizable Gesture Names:** A user-friendly interface to add, rename, and delete gesture classes directly from the GUI. Your custom gesture set is saved and persists across application restarts.
- **Improved Camera View:** The camera feed in both the data collection and inference pages is now scaled to fit the window, providing a wider, more natural field of view.
- **Enhanced Stability:** Implemented critical fixes to prevent memory-related crashes and ensure the UI remains robust and responsive, especially during gesture editing.

## 4. Workflow

1.  **🚀 Initial Setup**: Run `./start_app.sh` to automatically build all components and launch the GUI.
2.  **📊 Data Collection**: Use the **Data Collection** tab to record temporal gestures. Each new recording is automatically appended to a consolidated CSV file for that gesture, located at `models/data/{gesture_name}/{gesture_name}.csv`. This simplifies data management.
3.  **🏋️ Model Training**: Navigate to the **Training** tab and click "Start Training." This invokes the `train_c` executable, which now dynamically loads all user-defined gestures from the GUI configuration. It reads the consolidated CSV data, runs the training process for **150 epochs**, and saves the final `c_model.bin`.
4.  **⚛️ Model Quantization**: Navigate to the **Quantization** tab. This loads the trained floating-point model, converts its weights to 8-bit integers, and saves a new `c_model_quantized.bin` file. This step is crucial for optimizing the model for embedded deployment.
5.  **🎯 Real-time Inference**: Go to the **Inference** tab. Use the dropdown menu to select either the original `c_model.bin` or the `c_model_quantized.bin`. The C server will load the chosen model and perform real-time gesture recognition.

## 5. Model Quantization for Embedded Deployment

This project includes a complete, end-to-end workflow for **post-training quantization**. This process converts the model's 32-bit floating-point weights into highly efficient 8-bit integers (INT8), offering several key advantages for embedded systems like the Renesas RA8D1:

-   **Reduced Memory Footprint:** The quantized model is significantly smaller, consuming less of the valuable 1MB of SRAM.
-   **Faster Inference:** Integer arithmetic is computationally less expensive than floating-point math, enabling faster predictions.
-   **Energy Efficiency:** Reduced computational load leads to lower power consumption.

The quantization process is seamlessly integrated into the GUI, allowing you to convert a trained model with a single click. The C inference server is model-aware, meaning it can dynamically load and run inference with either the float or quantized model, making it easy to compare their performance.

## 5. System Architecture & Technical Specifications

The system employs a hybrid architecture, leveraging a Python-based GUI for user interaction and a high-performance C backend for model execution.

```
┌───────────────────┐      Persistent TCP     ┌───────────────────┐
│   Python Frontend │ ◄─────────────────────► │    C Backend      │
│ (PyQt6 GUI)       │         Socket          │ (Inference Engine)│
├───────────────────┤                         ├───────────────────┤
│ - Data Collection │                         │ - TCN Forward Pass│
│ - Training Control│                         │ - Model Loading   │
│ - Live Inference  │                         │ - Diagnostics     │
│ - MediaPipe       │                         │ - Static Memory   │
└───────────────────┘                         └───────────────────┘
         │                                             ▲
         ▼                                             │
┌───────────────────┐       C Training        ┌───────────────────┐
│  Training Data    │        Process          │   Binary Model    │
│   (CSV Files)     │ ──────────────────────► |  (c_model.bin)    │
└───────────────────┘                         └───────────────────┘
```

### Components

| Component | Technology | Role |
| :--- | :--- | :--- |
| **GUI Frontend** | Python (PyQt6) | Manages user workflow, data collection, and visualization. |
| **Backend Engine** | C | Handles all ML computation (training and inference) with static memory. |
| **Communication** | TCP Sockets | Persistent connection for low-latency binary data streaming. |

### Model & Data Pipeline

| Aspect | Specification |
| :--- | :--- |
| **Model Type** | Custom Temporal Convolutional Network (TCN) with **8 channels**. |
| **Input Shape** | 20 frames × **63 features** (21 landmarks × 3 coordinates). |
| **Activation** | Leaky ReLU (α=0.01) to prevent dying neurons. |
| **Optimizer** | Adam with He Initialization. |
| **Temporal Sampling**| Stride-5 for both training and inference to ensure consistency. |
| **Memory Footprint** | < 1MB SRAM (Static allocation, verified at compile-time). |
| **Performance** | >98% accuracy; >30 FPS inference speed. |

## 5. Project Structure

```
.
├── README.md                    # This comprehensive documentation
├── start_app.sh                 # Master startup script with process management
│
├── RA8D1_Simulation/            # C Backend Implementation
│   ├── main.c                   # TCP inference server (float/quantized)
│   ├── train_in_c.c             # Training executable main
│   ├── quantize.c               # Quantization executable main
│   ├── training_logic.c/h       # Core TCN implementation (float/quantized)
│   ├── mcu_constraints.h        # RA8D1 memory constraints and compile-time checks
│   └── Makefile                 # Build system for C executables
│
├── gui_app/                     # Python GUI Application
│   ├── main_app.py              # Main PyQt6 application with 4-page navigation
│   ├── logic.py                 # Core classes: HandTracker, GesturePredictor
│   └── ... pages ...            # Individual GUI pages for each workflow stage
│
├── RA8D1_Simulation/            # (Continued)
│   └── c_model_quantized.bin    # Quantized INT8 model output
│
├── models/                      # Data and Model Storage
│   ├── data/                    # Training data organized by gesture class
│   └── c_model.bin              # Trained FP32 model in binary format
│
└── docs/                        # Project Documentation
    ├── explanation.md           # Detailed technical explanation
    └── struggles.md             # Development challenges and solutions
```

