# Temporal Hand Gesture Recognition for Renesas RA8D1

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://github.com/WillForEternity/PerClassLoRA--RA8D1)
[![Platform](https://img.shields.io/badge/Platform-Renesas%20RA8D1-blue)](https://www.renesas.com/us/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ra8d1-480-mhz-arm-cortex-m85-based-microcontroller-helium-and-trustzone)
[![Memory](https://img.shields.io/badge/Memory-%3C1MB%20SRAM-orange)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-98%25%2B-success)]()

## 1. Project Overview

This project delivers a production-ready, real-time temporal hand gesture recognition system optimized for the Renesas RA8D1 microcontroller. It features a custom-built Temporal Convolutional Network (TCN) designed to operate within a strict 1MB SRAM memory constraint, demonstrating the successful deployment of advanced machine learning on a resource-constrained embedded platform.

The system is fully functional, having resolved critical data pipeline discrepancies between training and inference environments. It now achieves high accuracy (>98%) and stable performance (>30 FPS) in real-world conditions.

## 2. System Architecture

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
┌───────────────────┐       C Training          ┌───────────────────┐
│  Training Data    │        Process            │   Binary Model    │
│   (CSV Files)     │ ───────────────────────►  |  (c_model.bin)    │
└───────────────────┘                           └───────────────────┘
```

## 3. Technical Specifications

### Model & Data Pipeline
| **Component** | **Specification** |
| :--- | :--- |
| **Model Type** | Custom Temporal Convolutional Network (TCN) |
| **TCN Channels** | 2 (Ultra-lean design) |
| **Kernel Size** | 3 |
| **Sequence Length** | 20 frames |
| **Input Features** | 60 (20 landmarks × 3 coordinates) |
| **Output Classes** | 3 (wave, swipe_left, swipe_right) |
| **Activation** | Leaky ReLU |
| **Optimizer** | Adam with He Initialization |
| **Hand Tracking** | MediaPipe (21 landmarks) |
| **Normalization** | Wrist-relative positioning and scaling |
| **Temporal Sampling** | Stride-5 for both training and inference |
| **Communication** | TCP socket with length-prefixed binary float arrays |

### Performance Metrics
| **Metric** | **Value** |
| :--- | :--- |
| **Training Accuracy** | >98% (500 epochs) |
| **Inference Speed** | >30 FPS |
| **SRAM Footprint** | <1MB (Static allocation) |
| **Prediction Latency**| <33ms |
| **System Stability**| Production-ready with robust error handling |

## 4. Quick Start Guide

### Prerequisites

```bash
# Install Python dependencies
pip install PyQt6 mediapipe opencv-python numpy

# Install C compiler (macOS example)
xcode-select --install
```

### Execution

```bash
# 1. Clone the repository
git clone https://github.com/WillForEternity/PerClassLoRA--RA8D1.git
cd PerClassLoRA--RA8D1

# 2. Run the application
# This script compiles the C backend and launches the Python GUI
python run_app.py
```

### Workflow
1.  **Data Collection**: Use the GUI to record and save gesture sequences.
2.  **Training**: Initiate the C-based training process from the GUI to generate `c_model.bin`.
3.  **Inference**: Switch to the inference tab for live, real-time gesture recognition using the trained model.

### **2. Complete Workflow**
1. **📊 Data Collection**: Use GUI to record gesture sequences
2. **🏋️ Training**: Train the TCN model with collected data
3. **🎯 Inference**: Real-time gesture recognition with live camera

### **3. Advanced Usage**
```bash
# Manual C server compilation
cd RA8D1_Simulation
gcc -o inference_server main.c training_logic.c -lm

# Manual training
./train_in_c

# Manual inference server
./inference_server
```

## 📁 Project Structure

The system uses a hybrid architecture to combine ease of use with high performance:

### Python GUI Frontend (PyQt6)
-   **Role**: High-level orchestrator managing user workflow and data collection
-   **Components**: 4-page application (Setup, Data Collection, Training, Inference)
-   **Key Classes**:
    -   `HandTracker`: MediaPipe-based hand detection and landmark extraction
    -   `GesturePredictor`: Persistent TCP client for real-time inference
-   **Responsibilities**: Camera interaction, data visualization, process management
-   **No ML Computation**: All machine learning is delegated to the C backend

### C Backend Engine
Two standalone C executables handle the core ML tasks with static memory allocation:

#### Training Executable (`train_c`)
-   **Purpose**: Command-line utility for TCN model training
-   **Input**: CSV files organized by gesture class directories
-   **Output**: Binary model file (`c_model.bin`) for deployment
-   **Features**: Adam optimization, data augmentation via overlapping windows, train/validation split

#### Inference Server (`ra8d1_sim`)
-   **Purpose**: TCP server for real-time gesture prediction
-   **Protocol**: Persistent connection with binary data streaming
-   **Input**: 20-frame sequences of normalized hand landmarks (1200 floats)
-   **Output**: Class prediction index and confidence score
-   **Memory**: Fully static allocation for embedded compatibility

## Technical Specifications

### Model Architecture
-   **Type**: Temporal Convolutional Network (TCN)
-   **Input Shape**: 20 frames × 60 features (20 landmarks × 3 coordinates, excluding wrist)
-   **TCN Layer**: 2 channels, kernel size 3, causal dilated convolution
-   **Output Layer**: Dense layer with 3 classes (wave, swipe_left, swipe_right)
-   **Activation**: Leaky ReLU (α=0.01) to prevent dying neurons
-   **Total Parameters**: 371 parameters (ultra-lean design, ~1.5KB footprint)

### Training Configuration
-   **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=1e-8)
-   **Learning Rate**: 0.001
-   **Epochs**: 200
-   **Batch Size**: 1 (online learning)
-   **Data Split**: 80% training, 20% validation
-   **Initialization**: He initialization for weights

### Memory Constraints
-   **Target Hardware**: Renesas RA8D1 MCU
-   **SRAM Budget**: 1MB (1,048,576 bytes)
-   **Flash Budget**: 2MB (2,097,152 bytes)
-   **Compile-time Verification**: Static assertions ensure memory compliance
-   **Allocation Strategy**: 100% static, zero dynamic allocation

## Getting Started

### Prerequisites
-   A C compiler (e.g., `gcc` or `clang`)
-   `make`
-   **Python 3.11** (This specific version is required for GUI dependencies on macOS).

### Running the Application

The entire application is managed by a single script.

```bash
# This single command builds the C code, sets up the Python environment,
# and starts the C server and the GUI in the correct order.
./start_app.sh
```

The script handles all dependencies and ensures the C server is running before launching the GUI. It also gracefully shuts down all processes on exit.

## Workflow

### 1. Initial Setup
-   Run `./start_app.sh` to automatically build C executables and set up Python environments
-   The setup process creates separate virtual environments for tracking and training dependencies
-   Verifies Python 3.11 compatibility and installs required packages

### 2. Data Collection
-   **GUI Page**: Data Collection tab in the main application
-   **Process**: Record temporal gestures using live camera feed
-   **Sequence Length**: Each gesture is captured as a 20-frame sequence
-   **Data Format**: CSV files with normalized hand landmarks (60 features per frame)
-   **Organization**: Files saved in `models/data/{gesture_name}/` directories
-   **Normalization**: Wrist landmark used as origin, then excluded from feature vector

### 3. Model Training
-   **GUI Page**: Training tab with live log streaming
-   **Process**: Click "Start Training" to invoke the `train_c` executable
-   **Data Loading**: Automatic discovery and loading of all CSV files by gesture class
-   **Data Augmentation**: Overlapping windows with stride=5 for increased training samples
-   **Training Loop**: 200 epochs with Adam optimizer and validation monitoring
-   **Output**: Binary model file (`c_model.bin`) optimized for embedded deployment
-   **Monitoring**: Real-time training logs displayed in GUI console

### 4. Real-time Inference
-   **GUI Page**: Inference tab with live camera feed
-   **Server**: Persistent TCP connection to `ra8d1_sim` inference server
-   **Data Flow**: 
     1. Camera captures frames → MediaPipe extracts landmarks
     2. Landmarks normalized and buffered into 20-frame sequences
     3. Complete sequences sent to C server via TCP
     4. Server returns prediction class and confidence score
-   **Display**: Live gesture recognition with confidence visualization

## Complete File Structure

```
.
├── README.md                    # This comprehensive documentation
├── start_app.sh                 # Master startup script with process management
├── .gitignore                   # Git ignore patterns
├── simulation.log               # Training and inference logs
│
├── RA8D1_Simulation/            # C Backend Implementation
│   ├── main.c                   # TCP inference server main executable
│   ├── train_in_c.c             # Training executable main with hyperparameters
│   ├── training_logic.c         # Core TCN implementation (505 lines)
│   ├── training_logic.h         # TCN data structures and function declarations
│   ├── mcu_constraints.h        # RA8D1 memory constraints and compile-time checks
│   ├── Makefile                 # Build system for C executables
│   ├── ra8d1_sim*               # Compiled inference server executable
│   ├── train_c*                 # Compiled training executable
│   └── *.o                      # Object files from compilation
│
├── gui_app/                     # Python GUI Application
│   ├── main_app.py              # Main PyQt6 application with 4-page navigation
│   ├── logic.py                 # Core classes: HandTracker, GesturePredictor
│   ├── setup_page.py            # Environment setup and dependency management
│   ├── data_collection_page.py  # Live camera data collection interface
│   ├── training_page.py         # Training control with live log streaming
│   ├── inference_page.py        # Real-time gesture recognition interface
│   ├── requirements.txt         # GUI-specific Python dependencies
│   ├── venv_gui/                # Virtual environment for GUI
│   └── __pycache__/             # Python bytecode cache
│
├── Python_Hand_Tracker/         # Legacy Python ML Components
│   ├── hand_tracker.py          # Original MediaPipe hand tracking
│   ├── train_model.py           # Original TensorFlow model training
│   ├── inspect_model.py         # Model analysis utilities
│   ├── requirements_tracker.txt # Dependencies for hand tracking
│   ├── requirements_training.txt# Dependencies for TensorFlow training
│   ├── venv_tracker/            # Virtual environment for tracking
│   └── venv_training/           # Virtual environment for training
│
├── models/                      # Data and Model Storage
│   ├── data/                    # Training data organized by gesture class
│   │   ├── wave/                # Wave gesture CSV files
│   │   │   ├── wave_1.csv       # Individual gesture sequences
│   │   │   └── wave_*.csv       # Additional wave samples
│   │   ├── swipe_left/          # Left swipe gesture CSV files
│   │   └── swipe_right/         # Right swipe gesture CSV files
│   ├── c_model.bin              # Trained TCN model in binary format
│   ├── c_model_fixed.bin        # Backup/alternative model version
│   ├── model.onnx               # ONNX format model (legacy)
│   ├── saved_model/             # TensorFlow SavedModel format (legacy)
│   ├── training_history.csv     # Training metrics and loss curves
│   ├── dense_2_weights.bin      # Extracted layer weights (legacy)
│   └── dense_2_biases.bin       # Extracted layer biases (legacy)
│
└── docs/                        # Project Documentation
    ├── explanation.md           # Detailed technical explanation
    ├── PROGRESS_REPORT.md       # Development progress and milestones
    └── struggles.md             # Development challenges and solutions
```

### Key File Descriptions

#### Core Executables
- **`start_app.sh`**: Master script that handles C server startup, GUI launch, and cleanup
- **`ra8d1_sim`**: TCP inference server that loads the trained model and processes real-time predictions
- **`train_c`**: Training executable that processes CSV data and outputs the binary model

#### C Implementation Files
- **`training_logic.c/h`**: Complete TCN implementation with forward/backward passes, Adam optimizer
- **`main.c`**: Inference server with persistent TCP connection handling
- **`train_in_c.c`**: Training loop with data loading, validation, and model saving
- **`mcu_constraints.h`**: Memory budget definitions and compile-time assertions

#### Python GUI Components
- **`logic.py`**: Core classes for hand tracking and gesture prediction communication
- **`main_app.py`**: Main application window with navigation and page management
- **Page modules**: Individual GUI pages for each workflow stage

#### Data Organization
- **CSV Format**: Each file contains a single gesture sequence (20 frames × 60 features)
- **Binary Model**: Optimized format for embedded deployment with minimal overhead
- **Class Directories**: Gesture types organized in separate folders for automatic discovery
