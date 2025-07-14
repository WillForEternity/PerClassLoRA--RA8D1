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

To collect training data, activate the **tracker** environment and run the `hand_tracker.py` script in collection mode:

```bash
# Activate the tracker environment
source Python_Hand_Tracker/venv_tracker/bin/activate

# Run the script in data collection mode
python Python_Hand_Tracker/hand_tracker.py --mode collect
```

The script will guide you through collecting samples for several gestures (fist, palm, pointing). Follow the on-screen prompts. The data will be saved automatically to the `models/data/` directory.

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

### 4. Run the End-to-End Simulation

This step demonstrates the live inference pipeline from the Python hand tracker to the C simulation. It requires two separate terminals.

**Terminal 1: Start the C Simulation Server**

First, compile and run the C application. This will act as a persistent server, waiting for the Python client to connect and performing inference on the incoming data.

```bash
# Navigate to the simulation directory
cd RA8D1_Simulation

# Compile the C code, linking the ONNX Runtime library
# (Note: The path to onnxruntime may vary depending on your installation)
gcc main.c -o simulation -I/opt/homebrew/opt/onnxruntime/include -L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime

# Run the simulation server
./simulation
```

The server will start and print `Waiting for a client connection...`.

**Terminal 2: Start the Python Hand Tracker Client**

In a new terminal, activate the **tracker** environment and run the hand tracker in inference mode. It will connect to the C server and begin streaming data.

```bash
# Activate the tracker environment
source Python_Hand_Tracker/venv_tracker/bin/activate

# Run the hand tracker client
python Python_Hand_Tracker/hand_tracker.py --mode inference
```

Once connected, you will see the live camera feed. The C server terminal will now print the live gesture predictions (e.g., `Prediction: fist`) as it receives and analyzes the landmark data.
