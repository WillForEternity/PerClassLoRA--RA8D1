# Final Project Report: Temporal Gesture Recognition for Renesas RA8D1

## Project Status: 🎉 COMPLETE & FULLY FUNCTIONAL! 🎉

**Final Update (July 16, 2025)**: The project has been successfully transformed into a **temporal gesture recognition system**. The initial static gesture model was replaced with a **Temporal Convolutional Network (TCN)**, and the entire system—from data collection to C-based training and inference—is now stable and feature-complete.

---

## Summary of Accomplishments

The project delivered a robust, end-to-end solution for building and testing a custom temporal gesture model for an embedded target. The final application provides a seamless experience for managing the entire ML workflow.

### Key Features Delivered:

-   **✅ Temporal Data Pipeline**: The GUI and C backend now support collecting, processing, and training on sequences of hand landmarks, enabling the recognition of dynamic gestures like "wave" and "swipe."

-   **✅ Temporal Convolutional Network (TCN) in C**: A state-of-the-art TCN was implemented from scratch in pure C for the backend. This includes all necessary layers, activation functions (Leaky ReLU), and a stable backpropagation implementation.

-   **✅ Unified GUI Application**: A single, intuitive PyQt6 interface for the entire ML pipeline.

-   **✅ Live Training Log Viewer**: The training page was enhanced with a real-time log console that streams output directly from the C training executable, providing immediate feedback on epoch loss and progress.

-   **✅ Real-Time Temporal Inference**: The inference page streams camera data to the C server, which now uses the trained TCN to recognize and display predictions for temporal gestures.

-   **✅ Embedded-Ready C Backend**: The entire C backend operates with **100% static memory allocation** and includes compile-time checks to ensure the model's memory footprint does not exceed the 1MB SRAM limit of the Renesas RA8D1.

-   **✅ Automated Process Management**: A robust `start_app.sh` script orchestrates the C inference server and Python GUI, eliminating race conditions and ensuring a reliable, one-command startup.

-   **✅ Rock-Solid Communication Protocol**: The entire server-client communication layer was overhauled to fix critical stability bugs, resulting in a crash-free experience. This included implementing a persistent TCP connection and a buffered reader to correctly handle data streams.

### Core Technical Achievements

1.  **Successfully Ported to a Temporal Model**: The architecture was fundamentally refactored from a simple, static model to a complex TCN capable of understanding time-series data.
2.  **Stabilized C-Based Training**: Overcame the critical `NaN` loss bug by identifying the "Dying ReLU" problem and switching to Leaky ReLU, resulting in a stable and effective training process in C.
3.  **Rock-Solid GUI-Backend Integration**: Overcame critical stability bugs by re-architecting the server-client communication protocol. This involved fixing connection overload, race conditions, and data corruption issues, resulting in a truly robust and seamless user experience.
4.  **Data Pipeline Optimization**: Refined the input feature set by removing the redundant wrist landmark after normalization. This reduced the feature vector from 63 to 60 dimensions, which directly contributed to a lower training loss and a more accurate final model.

This represents the **complete realization** of the original embedded ML simulation concept, but elevated to handle the more complex and powerful domain of temporal gesture recognition.

---

## Final State

The project is feature-complete. The application is stable, the documentation has been updated to reflect the new temporal architecture, and the repository contains all necessary code and instructions to run the tool successfully.
