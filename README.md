# Hand Gesture Recognition GUI

This project provides a complete, user-friendly graphical interface (GUI) for building, training, and deploying a hand gesture recognition model. The application guides the user through the entire machine learning workflow, from setting up the environment to real-time inference.

 <!-- Replace with a real screenshot -->

## Features

- **End-to-End Workflow:** A single application for data collection, model training, and live inference.
- **User-Friendly Interface:** Built with PyQt6 for a clean and intuitive user experience.
- **Automated Setup:** The application checks for necessary dependencies and environments, guiding the user through the setup process.
- **Live Camera Feed:** Visual feedback during data collection and inference.
- **Real-Time Training Graph:** Monitor model performance with a live-updating graph during training.
- **Isolated Environments:** Uses separate Python virtual environments for the GUI, data collection, and training to prevent dependency conflicts.

## Getting Started

### 1. Prerequisites

- Python 3.11

### 2. Initial Setup

Before launching the application for the first time, you need to create the required virtual environments and install the dependencies.

```bash
# Create the virtual environments for the GUI and training
# NOTE: Using a different version of Python may cause installation errors.
python3.11 -m venv gui_app/venv_gui
python3.11 -m venv Python_Hand_Tracker/venv_training

# Install dependencies for the GUI
source gui_app/venv_gui/bin/activate
pip install -r gui_app/requirements_gui.txt
deactivate

# Install dependencies for training
source Python_Hand_Tracker/venv_training/bin/activate
pip install -r Python_Hand_Tracker/requirements_training.txt
deactivate
```

### 3. Running the Application

Once the setup is complete, launch the GUI application:

```bash
# Activate the GUI environment
source gui_app/venv_gui/bin/activate

# Run the main application
python -m gui_app.main_app
```

## Application Workflow

The GUI is organized into four distinct pages:

1.  **Setup:** Automatically checks if the required virtual environments and dependencies are correctly installed. It provides instructions if any part of the setup is missing.
2.  **Data Collection:** Use your camera to record hand gestures. Select a gesture to record, and the application will capture hand landmark data and save it to a CSV file.
3.  **Training:** Train the neural network on the data you collected. The application runs the training script in an isolated environment and displays a live graph of the model's accuracy.
4.  **Inference:** Test your trained model in real-time. The application displays a live camera feed and uses the model to predict and display your hand gestures.

## File Structure

```
.
├── gui_app/
│   ├── venv_gui/              # Virtual environment for the GUI
│   ├── main_app.py            # Main entry point for the application
│   ├── logic.py               # Core logic for HandTracking and Prediction
│   ├── setup_page.py          # UI and logic for the Setup page
│   ├── data_collection_page.py# UI and logic for Data Collection page
│   ├── training_page.py       # UI and logic for the Training page
│   ├── inference_page.py      # UI and logic for the Inference page
│   └── requirements_gui.txt   # Pip requirements for the GUI
│
├── Python_Hand_Tracker/
│   ├── venv_training/         # Virtual environment for model training
│   ├── train_model.py         # Script to train the model and export to ONNX
│   └── requirements_training.txt # Pip requirements for training
│
└── models/
    ├── data/                  # CSV files with hand landmark data
    ├── saved_model/           # TensorFlow SavedModel format
    └── model.onnx             # Final model in ONNX format
```
