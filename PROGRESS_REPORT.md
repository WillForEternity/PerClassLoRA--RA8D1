# Project Progress Report: Per-Class LoRA on RA8D1

## Project Goal

To design and simulate a complete adaptive machine learning workflow on a resource-constrained microcontroller (Renesas RA8D1). This involves offline training of a base model, deployment to a simulated MCU, and on-device adaptation using Per-Class LoRA.

---

## Current Status: Phase 2 Complete

We have successfully completed the first two major phases of the project. The primary blocker related to dependency conflicts has been resolved, and we now have a stable, reproducible workflow for training and exporting the model.

### Key Accomplishments

- **✅ Phase 1: Environment Setup & Data Collection (Complete)**
  - A Python-based hand tracker using `MediaPipe` was developed to stream hand landmark data.
  - A data collection mode was implemented to capture and save labeled gesture data into `.csv` files.
  - A C-based client was created to simulate the MCU, receiving data from the Python tracker via TCP sockets.

- **✅ Phase 2: Offline Model Training & Conversion (Complete)**
  - **Resolved Critical Dependency Conflicts:** Overcame a series of complex dependency issues involving `tensorflow`, `tf2onnx`, `numpy`, and `protobuf` by establishing a stable Python 3.11 environment with carefully pinned library versions.
  - **Successful Model Training:** A neural network was successfully trained on the collected gesture data, achieving high accuracy.
  - **Successful ONNX Export:** The trained TensorFlow model was successfully converted to the `model.onnx` format, which is now ready for deployment in the C/C++ simulation.

### Project Unblocked

The successful generation of the `model.onnx` file marks a major milestone. The project is no longer blocked by environment or dependency issues.

---

## Next Steps: Phase 3 - Simulation and Adaptation

With the base model ready, the next phase focuses on integrating it into the simulation and implementing the adaptive learning mechanism.

- **Integrate ONNX Runtime:** The C simulation will be updated to include the ONNX runtime, allowing it to load `model.onnx` and perform inference on the incoming hand landmark data.

- **Implement Per-Class LoRA:** The core adaptive learning mechanism will be implemented. This involves adding the LoRA layers to the model and developing the logic to update their weights based on new, user-specific data.

- **Simulate On-Device Learning:** The full feedback loop will be simulated, where the model adapts to new gestures in real-time within the constraints of the MCU environment.

- **Memory and Performance Auditing:** The memory footprint and performance of the simulation will be continuously audited against the `mcu_constraints.h` to ensure it remains within the target device's limits.
