#include "training_logic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h> // For directory traversal

// --- Private Helper Functions ---

// Helper to initialize weights using He initialization (good for ReLU)
static void initialize_weights(float* weights, size_t num_weights, size_t fan_in) {
    float std_dev = sqrtf(2.0f / fan_in);
    for (size_t i = 0; i < num_weights; ++i) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        const float epsilon = 1e-9;
        float z = sqrtf(-2.0f * logf(u1 + epsilon)) * cosf(2.0f * M_PI * u2);
        weights[i] = z * std_dev;
    }
}

// --- Model Initialization and Cleanup ---

void init_model(Model* model) {
    srand(time(NULL));

    // Initialize TCN Block
    initialize_weights(model->tcn_block.weights, sizeof(model->tcn_block.weights)/sizeof(float), INPUT_SIZE * TCN_KERNEL_SIZE);
    memset(model->tcn_block.biases, 0, sizeof(model->tcn_block.biases));

    // Initialize Output Layer
    initialize_weights(model->output_layer.weights, sizeof(model->output_layer.weights)/sizeof(float), TCN_CHANNELS);
    memset(model->output_layer.biases, 0, sizeof(model->output_layer.biases));
}

void save_model(const Model* model, const char* file_path) {
    FILE* file = fopen(file_path, "wb");
    if (!file) { perror("Failed to open file for writing"); return; }
    fwrite(model, sizeof(Model), 1, file);
    fclose(file);
    printf("Model saved to %s.\n", file_path);
}

int load_model(Model* model, const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) { perror("Could not open model file"); return -1; }
    size_t read_count = fread(model, sizeof(Model), 1, file);
    fclose(file);
    if (read_count != 1) { fprintf(stderr, "Error reading model file\n"); return -1; }
    printf("Model loaded successfully from %s.\n", file_path);
    return 0;
}

// --- Activation Functions ---
// Using Leaky ReLU to prevent the 'Dying ReLU' problem
float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}

float leaky_relu_derivative(float x) {
    return x > 0 ? 1 : 0.01f;
}

static void softmax(float* input, float* output, size_t size) {
    float max_val = input[0];
    for (size_t i = 1; i < size; ++i) if (input[i] > max_val) max_val = input[i];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }
    for (size_t i = 0; i < size; ++i) output[i] /= sum_exp;
}

// --- Forward Pass ---
void forward_pass(Model* model, const float* input_sequence, int epoch, int sample_idx) {
    // 1. TCN Block (Causal Convolution -> Leaky ReLU)
    for (int out_c = 0; out_c < TCN_CHANNELS; ++out_c) {
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            float sum = model->tcn_block.biases[out_c];
            for (int in_c = 0; in_c < INPUT_SIZE; ++in_c) {
                for (int k = 0; k < TCN_KERNEL_SIZE; ++k) {
                    int input_t = t - (TCN_KERNEL_SIZE - 1) + k; // Causal padding
                    if (input_t >= 0) {
                        sum += input_sequence[input_t * INPUT_SIZE + in_c] * 
                               model->tcn_block.weights[out_c * (INPUT_SIZE * TCN_KERNEL_SIZE) + in_c * TCN_KERNEL_SIZE + k];
                    }
                }
            }
            model->tcn_block.output[out_c * SEQUENCE_LENGTH + t] = leaky_relu(sum);
        }
    }

    // 2. Global Average Pooling
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        float sum = 0.0f;
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            sum += model->tcn_block.output[c * SEQUENCE_LENGTH + t];
        }
        model->pooled_output[c] = sum / SEQUENCE_LENGTH;
    }

    // 3. Output Layer (Dense)
    float final_layer_output[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; ++i) {
        float sum = model->output_layer.biases[i];
        for (int j = 0; j < TCN_CHANNELS; ++j) {
            sum += model->pooled_output[j] * model->output_layer.weights[i * TCN_CHANNELS + j];
        }
        final_layer_output[i] = sum;
    }

    // 4. Softmax Activation
    softmax(final_layer_output, model->output_layer.output, NUM_CLASSES);
}

// --- Backward Pass ---
void zero_gradients(Model* model) {
    memset(model->tcn_block.grad_weights, 0, sizeof(model->tcn_block.grad_weights));
    memset(model->tcn_block.grad_biases, 0, sizeof(model->tcn_block.grad_biases));
    memset(model->output_layer.grad_weights, 0, sizeof(model->output_layer.grad_weights));
    memset(model->output_layer.grad_biases, 0, sizeof(model->output_layer.grad_biases));
}

void backward_pass(Model* model, const float* input_sequence, const int* target_labels, size_t batch_size, int epoch, int sample_idx) {
    // This function assumes batch_size = 1 for simplicity
    zero_gradients(model);
    int target_label = target_labels[0];

    // 1. Gradient of Loss w.r.t. Softmax Input (dL/dO)
    float grad_loss[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; ++i) {
        float target = (i == target_label) ? 1.0f : 0.0f;
        grad_loss[i] = model->output_layer.output[i] - target;
    }

    // 2. Backprop through Output Layer (Dense)
    float grad_pooled_output[TCN_CHANNELS] = {0};
    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < TCN_CHANNELS; ++j) {
            model->output_layer.grad_weights[i * TCN_CHANNELS + j] += grad_loss[i] * model->pooled_output[j];
            grad_pooled_output[j] += grad_loss[i] * model->output_layer.weights[i * TCN_CHANNELS + j];
        }
        model->output_layer.grad_biases[i] += grad_loss[i];
    }

    // 3. Backprop through Global Average Pooling
    float grad_tcn_output[TCN_CHANNELS * SEQUENCE_LENGTH];
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        float grad_dist = grad_pooled_output[c] / SEQUENCE_LENGTH;
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            grad_tcn_output[c * SEQUENCE_LENGTH + t] = grad_dist;
        }
    }

    // 4. Backprop through Leaky ReLU
    for (int i = 0; i < TCN_CHANNELS * SEQUENCE_LENGTH; ++i) {
        grad_tcn_output[i] *= leaky_relu_derivative(model->tcn_block.output[i]);
    }

    // 5. Backprop through TCN Convolution
    for (int out_c = 0; out_c < TCN_CHANNELS; ++out_c) {
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            float grad_before_activation = grad_tcn_output[out_c * SEQUENCE_LENGTH + t];
            model->tcn_block.grad_biases[out_c] += grad_before_activation;
            for (int in_c = 0; in_c < INPUT_SIZE; ++in_c) {
                for (int k = 0; k < TCN_KERNEL_SIZE; ++k) {
                    int input_t = t - (TCN_KERNEL_SIZE - 1) + k;
                    if (input_t >= 0 && input_t < SEQUENCE_LENGTH) {
                        model->tcn_block.grad_weights[out_c * (INPUT_SIZE * TCN_KERNEL_SIZE) + in_c * TCN_KERNEL_SIZE + k] += grad_before_activation * input_sequence[input_t * INPUT_SIZE + in_c];
                    }
                }
            }
        }
    }
}


// --- Optimizer ---
void update_weights(Model* model, float learning_rate) {
    // --- Gradient Clipping ---
    float grad_norm_sq = 0.0f;
    const float clip_threshold = 1.0f; // A common value for clipping threshold

    // Calculate squared L2 norm for all gradients in the model
    for (size_t i = 0; i < sizeof(model->tcn_block.grad_weights)/sizeof(float); ++i) grad_norm_sq += model->tcn_block.grad_weights[i] * model->tcn_block.grad_weights[i];
    for (size_t i = 0; i < sizeof(model->tcn_block.grad_biases)/sizeof(float); ++i) grad_norm_sq += model->tcn_block.grad_biases[i] * model->tcn_block.grad_biases[i];
    for (size_t i = 0; i < sizeof(model->output_layer.grad_weights)/sizeof(float); ++i) grad_norm_sq += model->output_layer.grad_weights[i] * model->output_layer.grad_weights[i];
    for (size_t i = 0; i < sizeof(model->output_layer.grad_biases)/sizeof(float); ++i) grad_norm_sq += model->output_layer.grad_biases[i] * model->output_layer.grad_biases[i];

    float grad_norm = sqrtf(grad_norm_sq);

    // If the norm exceeds the threshold, scale all gradients
    if (grad_norm > clip_threshold) {
        float scale_factor = clip_threshold / grad_norm;
        for (size_t i = 0; i < sizeof(model->tcn_block.grad_weights)/sizeof(float); ++i) model->tcn_block.grad_weights[i] *= scale_factor;
        for (size_t i = 0; i < sizeof(model->tcn_block.grad_biases)/sizeof(float); ++i) model->tcn_block.grad_biases[i] *= scale_factor;
        for (size_t i = 0; i < sizeof(model->output_layer.grad_weights)/sizeof(float); ++i) model->output_layer.grad_weights[i] *= scale_factor;
        for (size_t i = 0; i < sizeof(model->output_layer.grad_biases)/sizeof(float); ++i) model->output_layer.grad_biases[i] *= scale_factor;
    }

    // --- Update Weights ---
    // Update TCN Block
    for (size_t i = 0; i < sizeof(model->tcn_block.weights)/sizeof(float); ++i) model->tcn_block.weights[i] -= learning_rate * model->tcn_block.grad_weights[i];
    for (size_t i = 0; i < sizeof(model->tcn_block.biases)/sizeof(float); ++i) model->tcn_block.biases[i] -= learning_rate * model->tcn_block.grad_biases[i];

    // Update Output Layer
    for (size_t i = 0; i < sizeof(model->output_layer.weights)/sizeof(float); ++i) model->output_layer.weights[i] -= learning_rate * model->output_layer.grad_weights[i];
    for (size_t i = 0; i < sizeof(model->output_layer.biases)/sizeof(float); ++i) model->output_layer.biases[i] -= learning_rate * model->output_layer.grad_biases[i];

    // NOTE: Gradients are now reset at the beginning of the backward pass for each sample.
}

// --- Data Loading with Overlapping Windows ---

#define MAX_FRAMES_PER_FILE 300 // Max frames in a single recording file
#define WINDOW_STRIDE 10        // Step size for the sliding window

int load_temporal_data(const char* dir_path, const char** gestures, int num_gestures, float** out_data, int** out_labels, int* out_num_sequences) {
    *out_num_sequences = 0;
    struct dirent *entry;
    char line[1024];

    // --- PASS 1: Count the total number of augmented sequences (windows) --- 
    for (int i = 0; i < num_gestures; ++i) {
        char gesture_path[256];
        snprintf(gesture_path, sizeof(gesture_path), "%s/%s", dir_path, gestures[i]);
        DIR *dp = opendir(gesture_path);
        if (!dp) { fprintf(stderr, "Could not open directory: %s\n", gesture_path); continue; }

        while ((entry = readdir(dp))) {
            if (strstr(entry->d_name, ".csv")) {
                char file_path[512];
                snprintf(file_path, sizeof(file_path), "%s/%s", gesture_path, entry->d_name);
                FILE* file = fopen(file_path, "r");
                if (!file) continue;

                int frame_count = 0;
                while (fgets(line, sizeof(line), file)) frame_count++;
                fclose(file);

                if (frame_count >= SEQUENCE_LENGTH) {
                    int num_windows = 1 + (frame_count - SEQUENCE_LENGTH) / WINDOW_STRIDE;
                    *out_num_sequences += num_windows;
                }
            }
        }
        closedir(dp);
    }

    if (*out_num_sequences == 0) { fprintf(stderr, "No valid sequences found to generate windows.\n"); return -1; }
    printf("Found %d raw gesture files. Generating %d training sequences with a stride of %d.\n", *out_num_sequences, *out_num_sequences, WINDOW_STRIDE);

    // --- Allocate memory for all augmented sequences ---
    *out_data = (float*)malloc(*out_num_sequences * SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
    *out_labels = (int*)malloc(*out_num_sequences * sizeof(int));
    if (!*out_data || !*out_labels) { fprintf(stderr, "Failed to allocate memory for augmented data\n"); return -1; }

    // --- PASS 2: Read data and create overlapping windows ---
    int current_sequence_idx = 0;
    float temp_frames[MAX_FRAMES_PER_FILE * INPUT_SIZE];

    for (int i = 0; i < num_gestures; ++i) {
        char gesture_path[256];
        snprintf(gesture_path, sizeof(gesture_path), "%s/%s", dir_path, gestures[i]);
        DIR *dp = opendir(gesture_path);
        if (!dp) continue;

        while ((entry = readdir(dp))) {
            if (strstr(entry->d_name, ".csv")) {
                char file_path[512];
                snprintf(file_path, sizeof(file_path), "%s/%s", gesture_path, entry->d_name);
                FILE* file = fopen(file_path, "r");
                if (!file) continue;

                // Read all frames from the file into a temporary buffer
                int frame_count = 0;
                while (fgets(line, sizeof(line), file) && frame_count < MAX_FRAMES_PER_FILE) {
                    char* token = strtok(line, ",");
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        if (token) {
                            temp_frames[frame_count * INPUT_SIZE + j] = atof(token);
                            token = strtok(NULL, ",");
                        } else {
                            temp_frames[frame_count * INPUT_SIZE + j] = 0.0f;
                        }
                    }
                    frame_count++;
                }
                fclose(file);

                // Generate overlapping windows from the temporary buffer
                if (frame_count >= SEQUENCE_LENGTH) {
                    for (int start_frame = 0; start_frame <= frame_count - SEQUENCE_LENGTH; start_frame += WINDOW_STRIDE) {
                        if(current_sequence_idx >= *out_num_sequences) {
                            fprintf(stderr, "Error: Exceeded allocated memory for sequences!\n");
                            break; // Avoid buffer overflow
                        }
                        // Copy the window into the final data array
                        memcpy(&((*out_data)[current_sequence_idx * SEQUENCE_LENGTH * INPUT_SIZE]),
                               &temp_frames[start_frame * INPUT_SIZE],
                               SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
                        (*out_labels)[current_sequence_idx] = i;
                        current_sequence_idx++;
                    }
                }
            }
        }
        closedir(dp);
    }
    *out_num_sequences = current_sequence_idx; // Update to the actual number of sequences created
    return 0; // Success
}

void shuffle_data(float* data, int* labels, int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;

        float temp_data[SEQUENCE_LENGTH * INPUT_SIZE];
        memcpy(temp_data, &data[i * SEQUENCE_LENGTH * INPUT_SIZE], SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
        memcpy(&data[i * SEQUENCE_LENGTH * INPUT_SIZE], &data[j * SEQUENCE_LENGTH * INPUT_SIZE], SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
        memcpy(&data[j * SEQUENCE_LENGTH * INPUT_SIZE], temp_data, SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
    }
}
