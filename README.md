# Per-Class LoRA on RA8D1 - Adaptive Learning Simulation

This project simulates an adaptive machine learning workflow for a resource-constrained microcontroller, the Renesas RA8D1. It demonstrates a complete pipeline from data collection and model training to deployment and simulated on-device learning.

The core idea is to train a base hand gesture recognition model, deploy it to a simulated MCU environment, and then use a lightweight technique like Per-Class LoRA (Low-Rank Adaptation) to adapt the model to new, user-specific gestures without requiring a full retrain.

## File Structure

```
/
├── Python_Hand_Tracker/         # Main Python scripts and virtual environments
│   ├── venv/                      # Virtual environment for the project
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

## Setup and Usage

This project uses two separate Python environments to manage conflicting dependencies.

### 1. Environment Setup

First, create a Python 3.11 virtual environment:

```bash
python3.11 -m venv Python_Hand_Tracker/venv
source Python_Hand_Tracker/venv/bin/activate
```

Next, install the dependencies for both tracking and training:

```bash
# Install tracker dependencies
pip install -r Python_Hand_Tracker/requirements_tracker.txt

# Install training dependencies
pip install -r Python_Hand_Tracker/requirements_training.txt
```

### 2. Data Collection

To collect training data for a new gesture, run the `hand_tracker.py` script in data collection mode. For example, to collect data for the 'fist' gesture:

```bash
python Python_Hand_Tracker/hand_tracker.py --mode collect --gesture fist
```

Hold your hand in the desired gesture in front of the camera. Press 'q' to quit and save the data to `models/data/fist.csv`.

### 3. Model Training

Once you have collected data for all your gestures, run the training script:

```bash
python Python_Hand_Tracker/train_model.py
```

This will:
1.  Load the `.csv` data from the `models/data` directory.
2.  Train a simple neural network on the data.
3.  Save the trained model to `models/saved_model`.
4.  Convert and save the final model to `models/model.onnx`.

### 4. Run the Simulation

First, compile and run the C-based MCU simulation:

```bash
cd RA8D1_Simulation
gcc main.c -o simulation
./simulation
```

Then, in a separate terminal, run the Python hand tracker in inference mode:

```bash
python Python_Hand_Tracker/hand_tracker.py --mode inference
```

The Python script will stream hand landmark data to the C application, which will receive and process it, simulating the on-device workflow.
