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

## Setup and Usage

This project uses two separate Python virtual environments to manage conflicting dependencies between the hand tracker (`mediapipe`) and the model training (`tensorflow`).

### 1. Environment Setup

First, create both virtual environments using Python 3.11:

```bash
# Create the environment for the hand tracker
python3.11 -m venv Python_Hand_Tracker/venv_tracker

# Create the environment for model training
python3.11 -m venv Python_Hand_Tracker/venv_training
```

Next, install the dependencies into their respective environments:

```bash
# Install tracker dependencies
./Python_Hand_Tracker/venv_tracker/bin/pip install -r Python_Hand_Tracker/requirements_tracker.txt

# Install training dependencies
./Python_Hand_Tracker/venv_training/bin/pip install -r Python_Hand_Tracker/requirements_training.txt
```

### 2. Data Collection

To collect training data for a new gesture, activate the **tracker** environment and run the `hand_tracker.py` script. For example, to collect data for the 'fist' gesture:

```bash
# Activate the tracker environment
source Python_Hand_Tracker/venv_tracker/bin/activate

# Run the script
python Python_Hand_Tracker/hand_tracker.py --mode collect --gesture fist
```

Hold your hand in the desired gesture in front of the camera. Press 'q' to quit and save the data to `models/data/fist.csv`.

### 3. Model Training

Once you have collected data, activate the **training** environment and run the training script:

```bash
# Activate the training environment
source Python_Hand_Tracker/venv_training/bin/activate

# Run the script
python Python_Hand_Tracker/train_model.py
```

This will:
1.  Load the `.csv` data from the `models/data` directory.
2.  Train a simple neural network on the data.
3.  Save the trained model to `models/saved_model`.
4.  Convert and save the final model to `models/model.onnx`.

### 4. Run the Simulation

This step requires two separate terminals.

**Terminal 1: Run the C Simulation**

```bash
cd RA8D1_Simulation
gcc main.c -o simulation
./simulation
```

**Terminal 2: Run the Python Hand Tracker**

Activate the **tracker** environment and run the `hand_tracker.py` script in inference mode:

```bash
# Activate the tracker environment
source Python_Hand_Tracker/venv_tracker/bin/activate

# Run the script in inference mode
python Python_Hand_Tracker/hand_tracker.py --mode inference
```

The Python script will stream hand landmark data to the C application, which will receive and process it, simulating the on-device workflow.
