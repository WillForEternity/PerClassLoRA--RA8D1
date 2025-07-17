# Temporal Hand Gesture Recognition for Embedded Systems

A complete, end-to-end system for training and deploying a **temporal hand gesture recognition** model on a resource-constrained embedded MCU (Renesas RA8D1). The project features a Python GUI for data management, a high-performance C backend that implements a **Temporal Convolutional Network (TCN)** with 100% static memory allocation, and an optimized data pipeline that improves model accuracy.

## Project Status: COMPLETE & STABLE

**Latest Update (July 16, 2025)**: The system is now **fully stable and reliable**. Critical bugs related to server-client communication have been resolved, resulting in a robust, crash-free user experience. The C-based TCN training is stable, and the Python GUI has been enhanced for a seamless workflow.

**Key Features:**
-   **Temporal Gesture Recognition**: Recognizes sequences of movements, not just static poses.
-   **Temporal Convolutional Network (TCN)**: A modern, efficient architecture for sequence modeling, implemented in pure C.
-   **Robust & Persistent Communication**: The GUI and C inference server now communicate over a stable, persistent TCP connection, eliminating previous crashes. A buffered reader on the client-side correctly handles TCP streaming, preventing data corruption.
-   **Python GUI**: A comprehensive PyQt6 application for data collection, training, and real-time inference.
-   **Live Training Logs**: The GUI's training page now streams log output directly from the C training process in real-time.
-   **Embedded-Ready C Backend**: All ML logic is in C with static memory allocation and compile-time memory checks to ensure it fits within the 1MB SRAM budget of the RA8D1 MCU.
-   **Optimized Data Pipeline**: The input feature set has been refined to 60 dimensions (20 landmarks x 3 coordinates), excluding the wrist landmark. This resulted in a more efficient model with significantly improved training accuracy.
-   **Automated Workflow**: A single script (`start_app.sh`) manages the C inference server and Python GUI, handling all setup and cleanup automatically.

## Architecture: Python Orchestrator, C Engine

The system uses a hybrid architecture to combine ease of use with high performance:

-   **Python GUI Frontend**: Manages the user workflow, including camera interaction, data collection, and visualization. It orchestrates the C backend but performs no ML calculations itself.
-   **C Backend Engine**: Two standalone C executables handle the core ML tasks:
    -   `train_c`: A command-line utility that trains the TCN model on collected gesture sequences.
    -   `ra8d1_sim`: A TCP server that loads the trained model and performs real-time inference on data streamed from the GUI.

## Getting Started

### Prerequisites
-   A C compiler (e.g., `gcc` or `clang`)
-   `make`
-   Python 3.11+

### Running the Application

An intelligent setup and startup script makes running the application simple.

1.  **First-Time Setup**: The GUI will automatically guide you through a one-time setup process that creates virtual environments, installs dependencies, and compiles the C code.
2.  **Start the App**: Use the main shell script to launch both the C inference server and the Python GUI.

```bash
# This single command starts the C server and the GUI
./start_app.sh
```

The script ensures the C server is running before launching the GUI and gracefully shuts down all processes on exit.

## Workflow

1.  **Data Collection**: Use the GUI to record temporal gestures (e.g., a 'wave' motion over 30 frames). Each recording is saved as a unique CSV file.
2.  **Training**: On the 'Training' page, click **"Start Training"**. The GUI invokes the `train_c` executable, which loads all CSV sequences, trains the TCN model, and saves the `c_model.bin` file. You can monitor the progress via the **live log console**.
3.  **Inference**: On the 'Inference' page, the GUI streams camera frames to the `ra8d1_sim` server, which returns real-time predictions for the recognized temporal gestures.

## File Structure

```
.
├── RA8D1_Simulation/       # C Backend Source Code
│   ├── train_in_c.c        # Main C code for training the TCN
│   ├── training_logic.c    # TCN model implementation (layers, backprop)
│   ├── main.c              # C code for the inference server
│   └── Makefile            # Compiles the C executables
│
├── gui_app/                # Python GUI Source Code
│   ├── main_app.py         # Main application entry point
│   ├── training_page.py    # UI and logic for the training page
│   └── ...
│
└── models/
    ├── data/
    │   ├── wave/           # Directory for 'wave' gesture sequences
    │   │   ├── wave_1.csv
    │   │   └── wave_2.csv
    │   └── swipe_left/     # Directory for 'swipe_left' sequences
    └── c_model.bin         # Final trained model in binary format
```
