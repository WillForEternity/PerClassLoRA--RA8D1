# Hand Gesture Recognition: C Backend Simulation

A complete hand gesture recognition system designed for deployment on the Renesas RA8D1 embedded MCU. This project features a Python GUI frontend that orchestrates a C backend implementing neural network training and inference with static memory allocation suitable for bare-metal embedded systems.

## Project Status: COMPLETE & WORKING!

**Latest Update (July 16, 2025)**: The entire system is now fully functional with seamless end-to-end operation!

**What Works:**
- Real-time hand gesture recognition (fist, palm, pointing)
- Complete Python GUI with data collection, training, and inference
- C backend neural network training and inference
- Automated startup script for hassle-free operation
- Robust socket communication between GUI and C server
- Memory-constrained C implementation ready for RA8D1 deployment

**Key Achievement**: Resolved all connection issues between Python GUI and C inference server through automated process management.

## Features

- **Python GUI Frontend:** A user-friendly interface built with PyQt6 for data collection and workflow management.
- **C-Based ML Backend:** All neural network training and inference logic is written in C, removing dependencies like TensorFlow and ONNX.
- **Static Memory Allocation:** The C model uses 100% static memory allocation, with no `malloc` or `free` calls for the model's data, making it suitable for bare-metal embedded systems.
- **Compile-Time Memory Check:** A `static_assert` in the C code ensures the model's SRAM usage does not exceed the hardware limits of the target MCU (1MB for the Renesas RA8D1), preventing runtime errors.
- **Hybrid Workflow:** The Python GUI seamlessly orchestrates the C executables, calling a `train_c` binary for training and communicating with a `ra8d1_sim` TCP server for inference.

## Getting Started

### 1. Prerequisites

- Python 3.11
- A C compiler (e.g., `gcc` or `clang`)
- `make`

### 2. Initial Setup

This setup involves creating a Python virtual environment for the GUI and compiling the C backend.

```bash
# 1. Create the virtual environment for the GUI
python3.11 -m venv gui_app/venv_gui

# 2. Install Python dependencies for the GUI
source gui_app/venv_gui/bin/activate
pip install -r gui_app/requirements_gui.txt
deactivate

# 3. Compile the C backend executables
cd RA8D1_Simulation/
make
cd ..
```

### 3. Running the Application

Once the setup is complete, use the automated startup script to launch both the C inference server and GUI application:

```bash
# Run the complete application (starts C server + GUI)
./start_app.sh
```

The script will:
- Start the C inference server in the background
- Wait for it to initialize and accept connections
- Launch the GUI application
- Automatically clean up both processes when you exit

**Alternative Manual Method:**
If you prefer to run components separately:

```bash
# Terminal 1: Start C inference server
cd RA8D1_Simulation/
./ra8d1_sim

# Terminal 2: Start GUI (in project root)
source gui_app/venv_gui/bin/activate
python gui_app/main_app.py
```

## Application Workflow

The GUI orchestrates the C backend as follows:

1.  **Data Collection:** Use your camera to record hand gestures. This part is handled in Python and saves data to CSV files.
2.  **Training:** When you click "Train Model," the GUI launches the compiled `./RA8D1_Simulation/train_c` executable as a subprocess. The C program handles data loading, model training, and saving the `c_model.bin` file.
3.  **Inference:** The GUI starts the `./RA8D1_Simulation/ra8d1_sim` executable, which acts as a TCP server. The Python application then sends hand landmark data to the C server over a socket and receives predictions back for display.

## File Structure

```
.
├── RA8D1_Simulation/
│   ├── main.c                 # C code for the inference server (ra8d1_sim)
│   ├── train_in_c.c           # C code for the training executable (train_c)
│   ├── training_logic.h       # Header for C neural network structures and functions
│   ├── training_logic.c       # C implementation of the neural network
│   ├── mcu_constraints.h      # Defines MCU memory limits for compile-time checks
│   └── Makefile               # Makefile to compile the C code
│
├── gui_app/
│   ├── venv_gui/              # Virtual environment for the GUI
│   ├── main_app.py            # Main entry point for the application
│   ├── logic.py               # Core GUI logic, including socket communication
│   └── ... (other UI files)
│
└── models/
    ├── data/                  # CSV files with hand landmark data
    └── c_model.bin            # The trained model, saved in a binary format by the C code
```
