#ifndef TRAINING_LOGIC_H
#define TRAINING_LOGIC_H

#include <stddef.h>

// Temporal Model Constants
#define NUM_LANDMARKS 21
#define INPUT_SIZE (NUM_LANDMARKS * 3)     // 21 landmarks * 3 coords
#define NUM_CLASSES 3           // Gestures
#define SEQUENCE_LENGTH 20      // Frames per sequence

// TCN Hyperparameters
#define TCN_CHANNELS 8         // TCN channels
#define TCN_KERNEL_SIZE 3

// TCN Data Structures (Static)

// Temporal Convolutional Network Block
typedef struct {
    // Causal, dilated conv
    float weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float biases[TCN_CHANNELS];
    float output[TCN_CHANNELS * SEQUENCE_LENGTH];

    // Gradients
    float grad_weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float grad_biases[TCN_CHANNELS];

    // Adam Optimizer state
    float m_weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float v_weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float m_biases[TCN_CHANNELS];
    float v_biases[TCN_CHANNELS];
} TCNBlock;

// Classification layer
typedef struct {
    float weights[NUM_CLASSES * TCN_CHANNELS];
    float biases[NUM_CLASSES];
    float output[NUM_CLASSES]; // Output probabilities

    // Gradients
    float grad_weights[NUM_CLASSES * TCN_CHANNELS];
    float grad_biases[NUM_CLASSES];

    // Adam Optimizer state
    float m_weights[NUM_CLASSES * TCN_CHANNELS];
    float v_weights[NUM_CLASSES * TCN_CHANNELS];
    float m_biases[NUM_CLASSES];
    float v_biases[NUM_CLASSES];
} OutputLayer;

// Complete TCN model
typedef struct {
    TCNBlock tcn_block;         // Single TCN block
    // Global Average Pooling output
    float pooled_output[TCN_CHANNELS];
    OutputLayer output_layer;

    // Loss gradient w.r.t. model output
    float loss_grad[NUM_CLASSES]; 
} Model;

// Lean inference-only model
// Must sync with Model struct
typedef struct {
    float weights[TCN_CHANNELS * INPUT_SIZE * TCN_KERNEL_SIZE];
    float biases[TCN_CHANNELS];
} InferenceTCNBlock;

typedef struct {
    float weights[NUM_CLASSES * TCN_CHANNELS];
    float biases[NUM_CLASSES];
} InferenceOutputLayer;

typedef struct {
    InferenceTCNBlock tcn_block;
    InferenceOutputLayer output_layer;
} InferenceModel;

// Compile-time SRAM check
#include <assert.h>
#include "mcu_constraints.h"

// Compile-time check for model size vs SRAM limit
// Note: Checks model size only, not data buffers.
static_assert(sizeof(Model) < APP_SRAM_LIMIT, "Error: Model size exceeds SRAM budget!");

// Function Prototypes

// Model Init/Cleanup
void init_model(Model* model);
void free_model(Model* model);
void save_model(const Model* model, const char* file_path);
int load_inference_model(InferenceModel* model, const char* file_path);

// Forward Pass
// Training forward pass (full model)
void forward_pass(Model* model, const float* input_data, int epoch, int sample_idx);

// Inference forward pass (lean model)
void forward_pass_inference(const InferenceModel* model, const float* input_data, float* final_output);

// Backward Pass
void backward_pass(Model* model, const float* input_data, const int* target_labels, size_t batch_size, int epoch, int sample_idx);

// Optimizer
void update_weights(Model* model, float learning_rate, float beta1, float beta2, float epsilon, int timestep);

// Data Loading/Preparation
int load_temporal_data(const char* dir_path, const char** gestures, int num_gestures, float** out_data, int** out_labels, int* out_num_sequences);
void split_data(int num_sequences, float train_split, int* train_indices, int* num_train, int* val_indices, int* num_val);
void shuffle_indices(int* indices, int num_samples);

#endif // TRAINING_LOGIC_H
