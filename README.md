# Temporal Hand Gesture Recognition for Embedded Systems

A complete, end-to-end system for training and deploying a **temporal hand gesture recognition** model on a resource-constrained embedded MCU (Renesas RA8D1). The project features a Python GUI for data management, a high-performance C backend that implements an **ultra-lean Temporal Convolutional Network (TCN)** with 100% static memory allocation, and an optimized data pipeline.

## Project Status: COMPLETE & STABLE

**Latest Update (July 17, 2025)**: The model has been successfully optimized to an **ultra-lean 2-channel TCN**, significantly reducing its memory footprint while maintaining high accuracy (~98%). A robust **sanity check** has been performed, confirming the model genuinely learns from data patterns, not noise. The entire training and inference pipeline is stable and validated.

## Key Features

### Core Capabilities
-   **Temporal Gesture Recognition**: Recognizes sequences of movements over time (20-frame sequences), not just static poses
-   **Ultra-Lean TCN Model**: A highly efficient 2-channel Temporal Convolutional Network with kernel size 3, implemented in pure C
-   **Real-time Inference**: Live camera feed processing with gesture prediction confidence scores
-   **Comprehensive GUI**: 4-page PyQt6 application covering setup, data collection, training, and inference

### Technical Achievements
-   **Static Memory Allocation**: 100% static memory usage with compile-time size verification for embedded deployment
-   **Advanced Optimization**: Adam optimizer with He initialization and Leaky ReLU activation to prevent dying neurons
-   **Validated Learning**: Model integrity confirmed via sanity check (label shuffling), proving genuine pattern learning
-   **Robust Communication**: Persistent TCP connection between GUI and C server with intelligent error handling
-   **Memory Constraint Compliance**: Strict 1MB SRAM budget enforcement for Renesas RA8D1 MCU compatibility
-   **Automated Workflow**: Single-script deployment with process management and cleanup

### Data Processing Pipeline
-   **Hand Landmark Detection**: MediaPipe-based 21-point hand tracking with 3D coordinates
-   **Intelligent Normalization**: Wrist-relative positioning with scale normalization and feature reduction (63→60 features)
-   **Temporal Windowing**: Overlapping sequence generation with configurable stride for data augmentation
-   **Binary Model Format**: Optimized model serialization for embedded deployment

## Architecture: Python Orchestrator, C Engine

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
