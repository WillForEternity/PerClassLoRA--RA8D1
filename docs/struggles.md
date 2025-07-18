# Project Struggles & Solutions Log

This document chronicles the major technical challenges encountered during the development of the temporal hand gesture recognition project, the debugging process, and the final solutions.

---

### 1. The `NaN` Bug: Stabilizing TCN Training in C

*   **Struggle:** This was the project's most critical bug. After refactoring the backend to a **Temporal Convolutional Network (TCN)**, the C training process became highly unstable, producing `NaN` (Not a Number) loss values within the first few samples.

*   **Debugging Journey & Solution:**
    1.  **Hypothesis 1 (Exploding Gradients):** We first implemented **gradient clipping** to cap the L2 norm of the gradients. This is a standard technique for RNNs and TCNs, but it did not solve the problem, pointing to a more fundamental issue.
    2.  **Hypothesis 2 (Incorrect Backprop):** We manually reviewed the backpropagation-through-time logic for the TCN's causal convolutions. While we found and fixed minor bugs, the `NaN` issue persisted.
    3.  **The Real Culprit (Dying ReLU):** Through extensive `printf` debugging, we traced the numerical values flowing through the network. The logs revealed that the gradients in the deeper layers of the TCN were consistently becoming `0.0`. The standard **ReLU activation function (`max(0, x)`)** was "killing" neurons, preventing gradients from flowing back. The learning load was pushed onto a few remaining neurons, whose weights would explode, causing the `NaN`.
    4.  **The Multi-Part Fix (Leaky ReLU & Adam):** The final solution required two key changes. First, we replaced ReLU with **Leaky ReLU (`f(x) = max(0.01*x, x)`)**, which ensures a small gradient always flows. Second, we replaced the basic SGD optimizer with **Adam**, which adapts the learning rate for each weight and provided a much more stable and effective training dynamic. This combination completely stabilized the training process.

---

### 2. Achieving Rock-Solid GUI-Backend Communication

*   **Struggle:** Even after solving the initial process orchestration, the system was plagued by critical, hard-to-debug stability issues during real-time inference. The C server would randomly crash, and the Python GUI would frequently log data parsing errors.

*   **Debugging Journey & Solution:**
    1.  **The "Broken Pipe" Mystery:** The C server would frequently die with a "Broken pipe" error. Extensive logging revealed the cause: the Python client was opening a new TCP connection for every single frame. This connection churn overwhelmed the server, causing it to crash.
    2.  **The Data Corruption Bug:** After fixing the connection churn, a more subtle bug appeared: the GUI would fail to parse responses from the server, throwing `ValueError: could not convert string to float`. This was a classic TCP streaming issue. The server was sending responses so quickly that the client's network buffer was receiving multiple messages at once (e.g., `response1\nresponse2\n`), which the naive parsing logic couldn't handle.
    3.  **The Multi-Pronged Solution:** A full, robust solution required fixing the entire communication protocol:
        -   **Persistent Connections:** Both the client and server were refactored to use a single, persistent TCP connection for the entire inference session.
        -   **Buffered Reading:** The Python client was modified to use a buffered reader (`socket.makefile('r')`), which correctly handles TCP streams by reading one newline-terminated response at a time.
        -   **Intelligent Startup:** The `start_app.sh` script was enhanced to be fully autonomous. It now detects and kills any zombie processes hogging the required port, preventing "Address already in use" errors and ensuring a clean start every time.

---

### 3. Model Overfitting and Validation

*   **Struggle:** After achieving a stable training pipeline, the model showed signs of overfitting and was too large for the target MCU. Furthermore, we needed to prove that its high accuracy was the result of genuine learning, not a subtle bug.

*   **Debugging Journey & Solution:**
    1.  **Aggressive Miniaturization:** The model's capacity was progressively reduced to combat overfitting and meet memory constraints. We iteratively lowered the TCN channels from 8, to 4, and finally to an **ultra-lean 2-channel architecture**. This minimal model successfully maintained high accuracy (~98%).
    2.  **Validating Learning (The Sanity Check):** To prove the model was truly learning, we needed to perform a sanity check. Initial attempts to shuffle data indices inside the C training loop led to memory corruption and segmentation faults.
    3.  **The Safe Sanity Check:** The successful solution was to implement the shuffle at a higher level. We modified the `load_temporal_data` function in C to **shuffle the label array** immediately after loading the data. This cleanly broke the data-label correlation without causing memory errors. Training on this shuffled data resulted in an accuracy of ~33% (random chance), proving the model learns real patterns.

---

### 4. Python Environment & Dependency Conflicts

*   **Struggle:** The initial project setup on Apple Silicon (M1/M2) was blocked by severe dependency conflicts with TensorFlow. Standard `pip install` commands failed.

*   **Solution:** The issue was resolved by creating a dedicated virtual environment with Python 3.11 and meticulously pinning the exact versions of critical packages in `requirements_training.txt`, including `tensorflow`, `tensorflow-metal`, and `tf2onnx`.

---

### 5. Correcting the Inference Data Pipeline

*   **Struggle:** The real-time inference was completely non-functional, providing nonsensical predictions. The root cause was a critical mismatch between the data being sent by the Python GUI and the data expected by the C inference server.

*   **Debugging Journey & Solution:**
    1.  **Code Review:** A thorough review of the Python `gui_app/logic.py` and the C `RA8D1_Simulation/main.c` files revealed the discrepancy. The Python `HandTracker` was extracting all 21 MediaPipe landmarks (including the wrist), while the C server's `forward_pass_inference` function was hard-coded to expect input data of exactly 20 landmarks (60 features per frame, with the wrist excluded).
    2.  **The Normalization Flaw:** The normalization logic was being applied *after* the flawed data was already buffered. The `GesturePredictor.predict` method was receiving 21 landmarks, attempting to normalize them (which itself was redundant), and then sending a malformed data packet to the C server.
    3.  **The Fix (Centralized Normalization):** The solution was to refactor the data processing pipeline to be correct-by-construction:
        -   The `HandTracker.get_landmark_data` function was modified to be the single source of truth for data preparation. It now immediately processes the raw 21 landmarks from MediaPipe, performs the normalization (subtracting the wrist as the origin), and returns a clean, flattened array of 60 floats (20 landmarks x 3 coordinates).
        -   The `GesturePredictor.predict` method was simplified to remove the redundant normalization step. It now directly buffers the pre-normalized 60-float arrays.
    4.  **Final Validation:** With this fix, the Python client now correctly assembles a sequence of 20 frames, each with 60 features, and sends a single, perfectly formatted packet of 1200 floats (`20 * 60`) to the C server, matching its exact expectation. This resolved the inference bug entirely.

---

## Final Outcome

All major blockers were resolved. The C-based TCN backend is numerically stable, adheres to embedded memory constraints, and communicates flawlessly with the Python frontend. The project successfully demonstrates a production-ready workflow for developing and deploying advanced temporal models on resource-constrained hardware.
