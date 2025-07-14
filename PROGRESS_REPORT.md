# Project Progress Report: Per-Class LoRA on RA8D1

## Project Goal

To design and simulate a complete adaptive machine learning workflow on a resource-constrained microcontroller (Renesas RA8D1). This involves offline training of a base model, deployment to a simulated MCU, and on-device adaptation using Per-Class LoRA.

---

## Current Status: Phase 3 In Progress

We have successfully established the end-to-end data pipeline between the Python hand tracker and the C simulation, marking a significant milestone in the project. The core infrastructure for live data streaming and processing is now complete.

### Key Accomplishments

- **✅ Phase 1: Environment Setup & Data Collection (Complete)**
  - A Python-based hand tracker using `MediaPipe` was developed.
  - A data collection mode was implemented to capture and save labeled gesture data.

- **✅ Phase 2: Offline Model Training & Conversion (Complete)**
  - **Resolved Critical Dependency Conflicts:** Overcame complex dependency issues by establishing a stable two-environment setup for tracking and training.
  - **Successful Model Training:** A neural network was successfully trained on the collected gesture data.
  - **Successful ONNX Export:** The trained TensorFlow model was successfully converted to the `model.onnx` format.

- **✅ Phase 3: Simulation and Adaptation (In Progress)**
  - **Established Robust C/Python Data Pipeline:**
    - Refactored the C simulation to act as a TCP server and the Python script as a client.
    - Implemented resilient connection logic to handle startup race conditions.
    - Successfully demonstrated live streaming of hand landmark data from the Python client to the C server.

### Project Unblocked

The successful data pipeline and the generated `model.onnx` file mean the project is fully unblocked and ready for the next stage of development.

---

## Next Steps: ONNX Integration

With the data pipeline in place, the immediate next step is to integrate the ONNX model into the C simulation.

- **Integrate ONNX Runtime:** The C simulation will be updated to include the ONNX Runtime C API. This involves downloading the library, updating the C code to load the model and run inference, and modifying the build process to link against it.

- **Implement Full Inference Pipeline:** The `main.c` loop will be updated to pass the received landmark data to the ONNX model, execute inference, and print the predicted gesture.

- **Future Work:** Once inference is working, we will proceed with implementing the Per-Class LoRA adaptation mechanism.
