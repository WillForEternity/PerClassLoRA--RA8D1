# Hand Gesture Recognition GUI

This project provides a complete, user-friendly graphical interface (GUI) for building and testing a hand gesture recognition model. The application guides the user through the entire machine learning pipeline, from initial setup and data collection to model training and real-time inference.

While originally conceived to simulate on-device learning for a Renesas RA8D1 MCU, the project has evolved into a powerful, general-purpose tool for creating and experimenting with custom gesture recognition models.

## Features

- **All-in-One Interface**: A single application manages the entire workflow, eliminating the need for command-line scripts.
- **Guided Setup**: An initial setup page verifies that all required dependencies are installed correctly.
- **Live Data Collection**: Collect training data for custom gestures using a live camera feed and progress indicators.
- **Real-Time Training Graph**: Monitor model performance during training with a live-updating graph of accuracy and validation accuracy.
- **Real-Time Inference**: Test the trained model with a live camera feed that displays the predicted gesture and confidence score in real-time.
- **Responsive UI**: Heavy tasks like training and inference run in separate threads to ensure the GUI remains responsive.

## File Structure

```
.
├── gui_app/                     # Source code for the PyQt6 GUI application
│   ├── venv_gui/                # Virtual environment for the GUI
│   ├── main_app.py              # Main entry point for the application
│   ├── setup_page.py            # UI and logic for the setup page
│   ├── data_collection_page.py  # UI and logic for data collection
│   ├── training_page.py         # UI and logic for model training
│   ├── inference_page.py        # UI and logic for real-time inference
│   ├── logic.py                 # Core logic for hand tracking and prediction
│   └── requirements_gui.txt     # Pip requirements for the GUI
│
├── Python_Hand_Tracker/         # Scripts and environment for model training
│   ├── venv_training/           # Virtual environment for model training
│   ├── train_model.py           # Script to train the neural network and export to ONNX
│   └── requirements_training.txt# Pip requirements for model training
│
├── models/                      # Trained models and training data
│   ├── data/                    # CSV files for hand landmark data
│   ├── saved_model/             # TensorFlow SavedModel format
│   └── model.onnx               # Final model in ONNX format
│
├── .gitignore
└── README.md
```

## Quick Start

Follow these steps to set up and run the application.

### 1. Prerequisites

- Python 3.11

### 2. Setup

First, create the Python virtual environments and install the required dependencies. This only needs to be done once.

```bash
# Create the virtual environments
# NOTE: Using a different version of Python may cause installation errors.
python3.11 -m venv gui_app/venv_gui
python3.11 -m venv Python_Hand_Tracker/venv_training

# Install dependencies for the GUI
source gui_app/venv_gui/bin/activate
pip install -r gui_app/requirements_gui.txt
deactivate

# Install dependencies for model training
source Python_Hand_Tracker/venv_training/bin/activate
pip install -r Python_Hand_Tracker/requirements_training.txt
deactivate
```

### 3. Running the Application

To launch the GUI, run the following command from the project root directory:

```bash
gui_app/venv_gui/bin/python -m gui_app.main_app
```

## How to Use the Application

The application will guide you through a four-step process:

1.  **Setup**: The initial page checks if all dependencies for both the GUI and training environments are correctly installed. Once all checks pass, you can proceed.

2.  **Data Collection**: Use this page to record examples for your hand gestures. Select a gesture, specify the number of samples to collect, and press "Start Collecting." Perform the gesture in front of your camera until the progress bar is full.

3.  **Training**: On this page, you can start the model training process. The application will use the data you collected to train a neural network. You can monitor the progress in the log window and watch the training/validation accuracy improve in real-time on the graph.

4.  **Inference**: After the model is trained, this page allows you to test it. A live camera feed will be displayed, and the model will predict your hand gesture in real-time, showing the result and its confidence level.
