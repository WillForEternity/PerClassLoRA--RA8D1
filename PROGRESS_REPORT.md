# Project Progress Report: Per-Class LoRA on RA8D1

## Project Goal

To design and simulate a complete adaptive machine learning workflow on a resource-constrained microcontroller (Renesas RA8D1). This involves offline training of a base model, deployment to a simulated MCU, and on-device adaptation using Per-Class LoRA.

---

## Current Status: Phase 3 Complete

We have successfully integrated the ONNX model into the C simulation and validated the full end-to-end inference pipeline. The system can now perform live gesture recognition by streaming data from the Python hand tracker to the C server, which runs the model and prints real-time predictions. This completes the core project infrastructure.

### Key Accomplishments

- **✅ Phase 1: Environment Setup & Data Collection (Complete)**
  - A Python-based hand tracker using `MediaPipe` was developed and used to collect training data.

- **✅ Phase 2: Offline Model Training & Conversion (Complete)**
  - **Resolved Dependency Conflicts:** Established a stable two-environment setup (Python 3.11) for tracking and training.
  - **Successful Model Training & Export:** Trained a neural network and exported it to the ONNX format.

- **✅ Phase 3: Simulation and Inference (Complete)**
  - **Established Robust C/Python Data Pipeline:** Built a resilient TCP client-server architecture for data streaming.
  - **Solved Inference Mismatch:** Diagnosed and fixed a critical bug where the model only predicted one class. This was resolved by retraining the model without feature scaling to match the raw data format used during inference.
  - **Debugged ONNX Runtime Integration:** Fixed a final crash by correcting the model's input tensor name (`keras_tensor:0` → `keras_tensor`) in the C code.
  - **Validated End-to-End System:** Confirmed that the C simulation correctly receives live data and produces varied, accurate gesture predictions.

### Project Unblocked

The successful end-to-end inference pipeline means the project is fully unblocked and ready for the final and most critical stage of development.

---

## Next Steps: Implement Per-Class LoRA

With the base model and inference pipeline working, the next step is to implement the core concept of the project: on-device adaptation.

- **Implement LoRA Matrix Operations:** Update the C code to simulate the application of LoRA matrices to the model's weights during inference. This will involve loading and applying the low-rank adaptation matrices.

- **Develop Adaptation Workflow:** Create a mechanism to simulate the "on-device" training of these LoRA matrices for a new, unseen gesture.

- **Evaluate Performance:** Measure the effectiveness of the adaptation and analyze the resource constraints (memory, computation) to ensure it's feasible for the target RA8D1 MCU.
