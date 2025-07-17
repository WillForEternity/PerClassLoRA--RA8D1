#ifndef TRAINING_LOGIC_H
#define TRAINING_LOGIC_H

#include <stddef.h>

// --- Constants for Temporal Model ---
#define INPUT_SIZE (20 * 3)     // 20 landmarks (excluding wrist), 3 coordinates per frame
#define NUM_CLASSES 3           // wave, swipe_left, swipe_right
#define SEQUENCE_LENGTH 100     // Number of frames per gesture sequence

// --- TCN Hyperparameters ---
#define TCN_CHANNELS 16         // Number of channels in TCN blocks
#define TCN_KERNEL_SIZE 3

// --- Data Structures for TCN (Static Allocation) ---

// A single block of a Temporal Convolutional Network
typedef struct {
    // Causal, dilated convolution
    float weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float biases[TCN_CHANNELS];
    float output[TCN_CHANNELS * SEQUENCE_LENGTH];

    // Gradients
    float grad_weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float grad_biases[TCN_CHANNELS];
} TCNBlock;

// Final dense layer for classification
typedef struct {
    float weights[NUM_CLASSES * TCN_CHANNELS];
    float biases[NUM_CLASSES];
    float output[NUM_CLASSES]; // Final class probabilities

    // Gradients
    float grad_weights[NUM_CLASSES * TCN_CHANNELS];
    float grad_biases[NUM_CLASSES];
} OutputLayer;

// Represents the complete TCN model
typedef struct {
    TCNBlock tcn_block;         // A single TCN block for simplicity
    // After TCN, we'll do a simple Global Average Pooling over the time dimension
    float pooled_output[TCN_CHANNELS];
    OutputLayer output_layer;

    // Gradient of the loss w.r.t. the final model output
    float loss_grad[NUM_CLASSES]; 
} Model;

// --- Compile-time SRAM check ---
#include <assert.h>
#include "mcu_constraints.h"

// This will cause a compile error if the Model size exceeds our SRAM budget
// Note: This only checks the model itself, not the full dataset buffer.
static_assert(sizeof(Model) < APP_SRAM_LIMIT, "Error: Model size exceeds SRAM budget!");

// --- Function Prototypes ---

// Model Initialization and Cleanup
void init_model(Model* model);
void free_model(Model* model);
void save_model(const Model* model, const char* file_path);
int load_model(Model* model, const char* file_path);

// Forward Pass
void forward_pass(Model* model, const float* input_data, int epoch, int sample_idx);

// Backward Pass (Backpropagation)
void backward_pass(Model* model, const float* input_data, const int* target_labels, size_t batch_size, int epoch, int sample_idx);

// Optimizer
void update_weights(Model* model, float learning_rate);

// Data Loading for Temporal Data
int load_temporal_data(const char* dir_path, const char** gestures, int num_gestures, float** out_data, int** out_labels, int* out_num_sequences);
void shuffle_data(float* data, int* labels, int num_samples);

#endif // TRAINING_LOGIC_H
