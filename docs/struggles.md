# Project Struggles & Solutions Log

This document chronicles the major technical challenges encountered during the development of the temporal hand gesture recognition project, the debugging process, and the final solutions.

---

### 1. The `NaN` Bug: Stabilizing TCN Training in C

*   **Struggle:** This was the project's most critical bug. After refactoring the backend to a **Temporal Convolutional Network (TCN)**, the C training process became highly unstable, producing `NaN` (Not a Number) loss values within the first few samples.

*   **Debugging Journey & Solution:**
    1.  **Hypothesis 1 (Exploding Gradients):** We first implemented **gradient clipping** to cap the L2 norm of the gradients. This is a standard technique for RNNs and TCNs, but it did not solve the problem, pointing to a more fundamental issue.
    2.  **Hypothesis 2 (Incorrect Backprop):** We manually reviewed the backpropagation-through-time logic for the TCN's causal convolutions. While we found and fixed minor bugs, the `NaN` issue persisted.
    3.  **The Real Culprit (Dying ReLU):** Through extensive `printf` debugging, we traced the numerical values flowing through the network. The logs revealed that the gradients in the deeper layers of the TCN were consistently becoming `0.0`. The standard **ReLU activation function (`max(0, x)`)** was "killing" neurons, preventing gradients from flowing back. The learning load was pushed onto a few remaining neurons, whose weights would explode, causing the `NaN`.
    4.  **The Fix (Leaky ReLU):** We replaced ReLU with **Leaky ReLU (`f(x) = max(0.01*x, x)`)**. This ensures a small, non-zero gradient always flows, even for negative inputs. This single change was the critical breakthrough that completely stabilized the training process.

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

### 3. Python Environment & Dependency Conflicts

*   **Struggle:** The initial project setup on Apple Silicon (M1/M2) was blocked by severe dependency conflicts with TensorFlow. Standard `pip install` commands failed.

*   **Solution:** The issue was resolved by creating a dedicated virtual environment with Python 3.11 and meticulously pinning the exact versions of critical packages in `requirements_training.txt`, including `tensorflow`, `tensorflow-metal`, and `tf2onnx`.

---

## Final Outcome

All major blockers were resolved. The C-based TCN backend is numerically stable, adheres to embedded memory constraints, and communicates flawlessly with the Python frontend. The project successfully demonstrates a production-ready workflow for developing and deploying advanced temporal models on resource-constrained hardware.
