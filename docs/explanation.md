# Project Explanation: Temporal Gesture Recognition with a C-Based TCN

This document outlines the final architecture of the Temporal Hand Gesture Recognition project. The system uses a Python GUI to orchestrate a C-based backend that implements a **Temporal Convolutional Network (TCN)**, simulating a complete workflow for a resource-constrained microcontroller (the Renesas RA8D1).

## Core Objective: High-Fidelity Temporal Model Simulation

The project's goal evolved from simulating a simple static model to implementing a sophisticated **temporal model** capable of recognizing dynamic gestures. The final architecture pairs a user-friendly Python GUI with a high-performance C backend that runs a TCN, all while adhering to the strict memory and processing constraints of an embedded system.

## Architectural Design

The hybrid architecture leverages the strengths of both Python and C:

-   **Python GUI (PyQt6):** Serves as the high-level orchestrator. It manages the user interface, camera input, and the collection of gesture **sequences** (e.g., 30 frames of a "wave" gesture). It does not perform any ML calculations.
-   **C Backend (TCN Engine):** All core ML logic is in C, compiled into two standalone executables:
    -   `train_c`: Trains the TCN model using the sequences of CSV files collected by the GUI.
    -   `ra8d1_sim`: A TCP server that loads the trained TCN and performs inference on gesture sequences streamed from the GUI.

### Temporal Data Workflow

The communication workflow is designed for time-series data:

1.  **Training**: The GUI invokes the `train_c` executable, which reads directories of CSV files (e.g., `models/data/wave/`), where each file represents one complete gesture sequence.
2.  **Inference**: The Python GUI buffers frames from the camera into a sequence of a fixed length (e.g., 30 frames). Once the buffer is full, the entire sequence is sent to the `ra8d1_sim` server via a TCP socket for a single prediction.

### Data Normalization and Feature Engineering

A key optimization in the data pipeline is the handling of the hand landmarks. For each frame:

1.  **Normalization**: The 21 hand landmarks are normalized by treating the wrist (landmark 0) as the origin (0,0,0). This makes the gesture data invariant to the hand's position in the camera frame.
2.  **Feature Exclusion**: The wrist landmark's own coordinates are **excluded** from the feature vector. Since it's always the origin, it provides no useful information to the model. This reduces the input feature size from 63 (21 * 3) to **60 (20 * 3)**, creating a more efficient and focused dataset that has been shown to improve model training performance.

## Key Technical Achievements

1.  **TCN in Pure C with Static Memory**: The project's primary achievement is the successful implementation of a TCN from scratch in C. All model parameters, activations, and gradients are stored in **statically allocated structs and arrays**. This complete avoidance of `malloc` for the model is critical for embedded systems, preventing memory leaks and ensuring predictable performance.

2.  **Compile-Time Memory Constraint**: To guarantee the model fits within the 1MB SRAM budget of the Renesas RA8D1, a `static_assert` in `mcu_constraints.h` checks the model's size at compile time. This provides immediate feedback if the network architecture becomes too large for the target hardware.

## Stabilizing the Server-Client Communication

While the C-based TCN was a major achievement, the initial system suffered from critical stability issues during real-time inference, leading to server crashes and data corruption. The root cause was a naive networking implementation.

1.  **The Problem: Connection Overload & Protocol Mismatch**: The Python GUI was opening a new TCP socket for every single frame sent to the inference server. This constant opening and closing of connections overwhelmed the C server, causing it to crash with "Broken pipe" errors. Furthermore, a protocol mismatch (the client expected a persistent connection while the server was closing it immediately) and a TCP streaming bug (multiple server responses were being read at once by the client) led to data parsing errors.

2.  **The Solution: Persistent Connections & Buffered Reading**: The fix required a complete overhaul of the communication protocol:
    -   **Persistent TCP Connection**: The Python client and C server were refactored to use a single, persistent TCP connection that remains open for the entire inference session. This eliminated the connection overhead and stabilized the server.
    -   **Buffered Reader**: On the client side, the socket was wrapped in a buffered reader (`makefile('r')`). This ensures that even if multiple responses arrive in the TCP stream at once, they are correctly queued and read one line at a time, fixing the data corruption bug.
    -   **Intelligent Startup**: The `start_app.sh` script was enhanced to detect and kill any zombie server processes occupying the required port, preventing "Address already in use" errors on startup.

## Debugging the C-Based TCN: The `NaN` Loss Problem

Stabilizing the C-based training was the most significant challenge. The model would consistently produce `NaN` (Not a Number) loss values, halting any learning. The debugging process revealed multiple compounding issues:

1.  **Initial Hypothesis (Exploding Gradients)**: Gradient clipping was implemented but did not solve the issue, suggesting a deeper problem.
2.  **The Real Culprit: "Dying ReLU"**: Through extensive logging, we discovered that the standard ReLU activation function (`max(0, x)`) was causing most neuron gradients to become zero. This "killed" the neurons, preventing weight updates and causing the remaining active neurons to receive pathologically large gradients, which resulted in `NaN` values.
3.  **The Solution: Leaky ReLU**: Switching to **Leaky ReLU** (`max(0.01*x, x)`) was the critical fix. It ensures a small, non-zero gradient always flows, preventing neurons from dying and stabilizing the network.
4.  **Correcting Backpropagation**: Further debugging uncovered subtle bugs in the TCN's backpropagation logic, which were corrected to ensure gradients flowed correctly through the temporal layers.

## Conclusion

By successfully evolving the project from a static model to a temporal one and solving the complex challenges of C-based TCN training, the project fully realizes its goal. The final application is a powerful, high-fidelity simulation of an advanced embedded AI system, demonstrating a production-ready pattern for deploying temporal models on resource-constrained devices.

**Status**: ✅ FULLY FUNCTIONAL - End-to-end temporal gesture recognition system operational.

