# Per-Class LoRA on RA8D1 - Adaptive Learning Simulation

This project simulates an adaptive machine learning workflow for a resource-constrained microcontroller, the Renesas RA8D1. It demonstrates a complete pipeline from data collection and model training to deployment and simulated on-device learning.

The core idea is to train a base hand gesture recognition model, deploy it to a simulated MCU environment, and then use a lightweight technique like Per-Class LoRA (Low-Rank Adaptation) to adapt the model to new, user-specific gestures without requiring a full retrain.

## File Structure

```
.
├── Python_Hand_Tracker/         # Main Python scripts and virtual environments
│   ├── venv_tracker/              # Virtual environment for the hand tracker
│   ├── venv_training/             # Virtual environment for model training
│   ├── hand_tracker.py          # Runs the MediaPipe hand tracker for inference or data collection
│   ├── requirements_tracker.txt # Pip requirements for the hand tracker (mediapipe, opencv)
│   ├── requirements_training.txt# Pip requirements for model training (tensorflow, tf2onnx)
│   └── train_model.py           # Script to train the neural network and export to ONNX
│
├── RA8D1_Simulation/            # C code simulating the MCU application
│   ├── main.c                   # Main application logic for the simulated MCU
│   └── mcu_constraints.h        # Defines memory constraints for the RA8D1
│
├── models/                      # Trained models and training data
│   ├── data/                    # CSV files containing collected hand landmark data
│   │   ├── fist.csv
│   │   ├── palm.csv
│   │   └── pointing.csv
│   ├── saved_model/             # TensorFlow SavedModel format of the trained model
│   └── model.onnx               # Final model in ONNX format for cross-platform use
│
├── .gitignore                   # Git ignore file
└── explanation.md               # Detailed project plan and phases
```

## Quick Start: Automated Workflow

This project includes an automation script, `run.sh`, to simplify the setup and execution of all major tasks. This script handles environment activation, C code compilation, and running the correct Python scripts for each stage of the workflow.

### 1. Initial Setup

Before running the script for the first time, you need to set up the Python virtual environments and install the required dependencies. This only needs to be done once.

```bash
# Create the virtual environments using Python 3.11
# NOTE: Using a different version of Python may cause installation errors.
python3.11 -m venv Python_Hand_Tracker/venv_tracker
python3.11 -m venv Python_Hand_Tracker/venv_training

# Install dependencies
./Python_Hand_Tracker/venv_tracker/bin/pip install -r Python_Hand_Tracker/requirements_tracker.txt
./Python_Hand_Tracker/venv_training/bin/pip install -r Python_Hand_Tracker/requirements_training.txt
```

### 2. Using the `run.sh` Script

After the initial setup, you can use the `run.sh` script with one of the following commands: `collect`, `train`, or `inference`.

**A. Collect Data**

To collect new training data, run:

```bash
./run.sh collect
```

The script will automatically activate the correct environment and launch the data collection interface. Follow the on-screen prompts to record gestures. Data is appended to the existing `.csv` files in `models/data/`.

**B. Train the Model**

After collecting data, train the model by running:

```bash
./run.sh train
```

This will activate the training environment, run the training script, and save the updated model to `models/model.onnx`.

**C. Run Inference**

To run the full end-to-end simulation, use the following command:

```bash
./run.sh inference
```

This command will:
1.  Compile the C simulation code (if not already compiled).
2.  Start the C simulation server in the background.
3.  Launch the Python hand tracker, which will connect to the server and begin streaming data.

The live camera feed will appear, and the terminal will show the real-time gesture predictions from the C simulation. To stop the simulation, simply close the camera window or press `Ctrl+C` in the terminal.
