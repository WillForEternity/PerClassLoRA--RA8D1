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

    // Initialize TCN Adam state
    memset(model->tcn_block.m_weights, 0, sizeof(model->tcn_block.m_weights));
    memset(model->tcn_block.v_weights, 0, sizeof(model->tcn_block.v_weights));
    memset(model->tcn_block.m_biases, 0, sizeof(model->tcn_block.m_biases));
    memset(model->tcn_block.v_biases, 0, sizeof(model->tcn_block.v_biases));

    // Initialize Output Layer
    initialize_weights(model->output_layer.weights, sizeof(model->output_layer.weights)/sizeof(float), TCN_CHANNELS);
    memset(model->output_layer.biases, 0, sizeof(model->output_layer.biases));

    // Initialize Output Layer Adam state
    memset(model->output_layer.m_weights, 0, sizeof(model->output_layer.m_weights));
    memset(model->output_layer.v_weights, 0, sizeof(model->output_layer.v_weights));
    memset(model->output_layer.m_biases, 0, sizeof(model->output_layer.m_biases));
    memset(model->output_layer.v_biases, 0, sizeof(model->output_layer.v_biases));
}

void save_model(const Model* model, const char* file_path) {
    FILE* fp = fopen(file_path, "wb");
    if (!fp) {
        perror("Error opening file for writing");
        return;
    }

    // --- Safe, Field-by-Field Writing ---
    // This prevents memory alignment issues by writing each array individually,
    // creating a packed file that the loader can read safely.
    // We only write the weights and biases, to match the InferenceModel format.

    // Write TCN block weights and biases
    fwrite(model->tcn_block.weights, sizeof(model->tcn_block.weights), 1, fp);
    fwrite(model->tcn_block.biases, sizeof(model->tcn_block.biases), 1, fp);

    // Write Output layer weights and biases
    fwrite(model->output_layer.weights, sizeof(model->output_layer.weights), 1, fp);
    fwrite(model->output_layer.biases, sizeof(model->output_layer.biases), 1, fp);
    
    // Diagnostic: Print what we're saving
    printf("[SAVE DIAGNOSTIC] Saving output layer weights:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  weight[%d]: %.6f\n", i, model->output_layer.weights[i]);
    }
    printf("  bias[0]: %.6f\n", model->output_layer.biases[0]);

    fclose(fp);
    printf("Model saved to %s in a safe, packed format.\n", file_path);
}

int load_inference_model(InferenceModel* model, const char* file_path) {
    FILE* fp = fopen(file_path, "rb");
    if (!fp) {
        perror("Failed to open model file for reading");
        return 0; // File not found or error
    }

    // --- Safe, Field-by-Field Reading ---
    // This prevents memory alignment issues by reading directly into each array,
    // respecting any padding the compiler might have added to the structs.

    // Read TCN block weights and biases
    size_t tcn_weights_read = fread(model->tcn_block.weights, sizeof(model->tcn_block.weights), 1, fp);
    size_t tcn_biases_read = fread(model->tcn_block.biases, sizeof(model->tcn_block.biases), 1, fp);

    // Read Output layer weights and biases
    size_t out_weights_read = fread(model->output_layer.weights, sizeof(model->output_layer.weights), 1, fp);
    size_t out_biases_read = fread(model->output_layer.biases, sizeof(model->output_layer.biases), 1, fp);

    fclose(fp);

    // Check that all read operations were successful
    int success = (tcn_weights_read == 1 && tcn_biases_read == 1 && 
                   out_weights_read == 1 && out_biases_read == 1);

    if (!success) {
        fprintf(stderr, "Error: Failed to read all components of the model file.\n");
    }

    return success;
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

// --- Forward Pass (Inference) ---
void forward_pass_inference(const InferenceModel* model, const float* input_data, float* final_output) {
    const int TCN_DILATION = 2; // TODO: This should be sourced from the header
    // Note: This function is simplified and does not store intermediate values
    // needed for backpropagation. It's for inference only.

    // --- TCN Block --- 
    float dilated_conv_output[SEQUENCE_LENGTH * TCN_CHANNELS] = {0};
    // Dilated Convolution
    for (int c_out = 0; c_out < TCN_CHANNELS; ++c_out) {
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            float sum = 0.0f;
            for (int k = 0; k < TCN_KERNEL_SIZE; ++k) {
                int t_in = t + (k - (TCN_KERNEL_SIZE - 1)) * TCN_DILATION;
                if (t_in >= 0 && t_in < SEQUENCE_LENGTH) {
                    for (int c_in = 0; c_in < INPUT_SIZE; ++c_in) {
                        sum += input_data[t_in * INPUT_SIZE + c_in] * model->tcn_block.weights[c_out * (INPUT_SIZE * TCN_KERNEL_SIZE) + c_in * TCN_KERNEL_SIZE + k];
                    }
                }
            }
            sum += model->tcn_block.biases[c_out];
            dilated_conv_output[t * TCN_CHANNELS + c_out] = leaky_relu(sum);
        }
    }

    // --- Global Average Pooling --- 
    float pooled_output[TCN_CHANNELS] = {0};
    for (int c = 0; c < TCN_CHANNELS; ++c) {
        float sum = 0.0f;
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            sum += dilated_conv_output[t * TCN_CHANNELS + c];
        }
        pooled_output[c] = sum / SEQUENCE_LENGTH;
    }

    // --- Output Layer --- 
    float output_logits[NUM_CLASSES];
    for (int j = 0; j < NUM_CLASSES; ++j) {
        output_logits[j] = 0;
        for (int i = 0; i < TCN_CHANNELS; ++i) {
            output_logits[j] += pooled_output[i] * model->output_layer.weights[j * TCN_CHANNELS + i];
        }
        output_logits[j] += model->output_layer.biases[j];
    }

    // --- Softmax --- 
    softmax(output_logits, final_output, NUM_CLASSES);
}

// --- Forward Pass (Training) ---
void forward_pass(Model* model, const float* input_sequence, int epoch, int sample_idx) {
    // 1. TCN Block (Causal Convolution -> Leaky ReLU)
    for (int out_c = 0; out_c < TCN_CHANNELS; ++out_c) {
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            double sum = model->tcn_block.biases[out_c]; // Use double for accumulator
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
    memset(final_layer_output, 0, sizeof(final_layer_output)); // CRITICAL: Initialize to zero
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
    // Note: We don't need to compute grad_input since this is the first layer.
    for (int out_c = 0; out_c < TCN_CHANNELS; ++out_c) {
        for (int k = 0; k < TCN_KERNEL_SIZE; ++k) {
            for (int in_c = 0; in_c < INPUT_SIZE; ++in_c) {
                float weight_grad = 0.0f;
                for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
                    int input_t = t - (TCN_KERNEL_SIZE - 1) + k; // This logic is for 'valid' convolution padding
                    if (input_t >= 0 && input_t < SEQUENCE_LENGTH) {
                        weight_grad += grad_tcn_output[out_c * SEQUENCE_LENGTH + t] * input_sequence[input_t * INPUT_SIZE + in_c];
                    }
                }
                model->tcn_block.grad_weights[out_c * (INPUT_SIZE * TCN_KERNEL_SIZE) + in_c * TCN_KERNEL_SIZE + k] += weight_grad;
            }
        }
        // Update bias gradient
        float bias_grad = 0.0f;
        for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
            bias_grad += grad_tcn_output[out_c * SEQUENCE_LENGTH + t];
        }
        model->tcn_block.grad_biases[out_c] += bias_grad;
    }
}


// --- Optimizer ---
void update_weights(Model* model, float learning_rate, float beta1, float beta2, float epsilon, int timestep) {
    // Bias correction terms
    float beta1_t = powf(beta1, timestep);
    float beta2_t = powf(beta2, timestep);
    // Ensure denominators are not zero
    if (1.0f - beta1_t == 0.0f) beta1_t -= 1e-9; 
    if (1.0f - beta2_t == 0.0f) beta2_t -= 1e-9;
    float lr_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

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
    // Update TCN block
    for (size_t i = 0; i < sizeof(model->tcn_block.weights) / sizeof(float); ++i) {
        float grad = model->tcn_block.grad_weights[i];
        model->tcn_block.m_weights[i] = beta1 * model->tcn_block.m_weights[i] + (1 - beta1) * grad;
        model->tcn_block.v_weights[i] = beta2 * model->tcn_block.v_weights[i] + (1 - beta2) * (grad * grad);
        model->tcn_block.weights[i] -= lr_t * model->tcn_block.m_weights[i] / (sqrtf(model->tcn_block.v_weights[i]) + epsilon);
    }
    for (size_t i = 0; i < TCN_CHANNELS; ++i) {
        float grad = model->tcn_block.grad_biases[i];
        model->tcn_block.m_biases[i] = beta1 * model->tcn_block.m_biases[i] + (1 - beta1) * grad;
        model->tcn_block.v_biases[i] = beta2 * model->tcn_block.v_biases[i] + (1 - beta2) * (grad * grad);
        model->tcn_block.biases[i] -= lr_t * model->tcn_block.m_biases[i] / (sqrtf(model->tcn_block.v_biases[i]) + epsilon);
    }

    // Update output layer
    for (size_t i = 0; i < sizeof(model->output_layer.weights) / sizeof(float); ++i) {
        float grad = model->output_layer.grad_weights[i];
        model->output_layer.m_weights[i] = beta1 * model->output_layer.m_weights[i] + (1 - beta1) * grad;
        model->output_layer.v_weights[i] = beta2 * model->output_layer.v_weights[i] + (1 - beta2) * (grad * grad);
        model->output_layer.weights[i] -= lr_t * model->output_layer.m_weights[i] / (sqrtf(model->output_layer.v_weights[i]) + epsilon);
    }
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        float grad = model->output_layer.grad_biases[i];
        model->output_layer.m_biases[i] = beta1 * model->output_layer.m_biases[i] + (1 - beta1) * grad;
        model->output_layer.v_biases[i] = beta2 * model->output_layer.v_biases[i] + (1 - beta2) * (grad * grad);
        model->output_layer.biases[i] -= lr_t * model->output_layer.m_biases[i] / (sqrtf(model->output_layer.v_biases[i]) + epsilon);
    }

    // Zero gradients after updating
    zero_gradients(model);
}

// --- Data Loading with Overlapping Windows ---



#define WINDOW_STRIDE 5 // Create a new window every 5 frames

int load_temporal_data(const char* dir_path, const char** gestures, int num_gestures, float** out_data, int** out_labels, int* out_num_sequences) {
    char path[1024];
    int total_sequences = 0;

    // --- First Pass: Count total possible sequences ---
    printf("Starting pass 1: Counting sequences...\n");
    for (int i = 0; i < num_gestures; i++) {
        snprintf(path, sizeof(path), "%s/%s", dir_path, gestures[i]);
        DIR *d = opendir(path);
        if (!d) {
            fprintf(stderr, "Warning: Could not open directory %s\n", path);
            continue;
        }
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            if (strstr(dir->d_name, ".csv") == NULL) continue;
            
            char file_path[2048];
            snprintf(file_path, sizeof(file_path), "%s/%s", path, dir->d_name);
            FILE* file = fopen(file_path, "r");
            if (!file) continue;

            int frame_count = 0;
            char line_buffer[4096];
            while (fgets(line_buffer, sizeof(line_buffer), file)) {
                frame_count++;
            }
            fclose(file);

            if (frame_count >= SEQUENCE_LENGTH) {
                total_sequences += (frame_count - SEQUENCE_LENGTH) / WINDOW_STRIDE + 1;
            }
        }
        closedir(d);
    }

    if (total_sequences == 0) {
        fprintf(stderr, "Error: No valid data sequences found. All files might be shorter than SEQUENCE_LENGTH (%d).\n", SEQUENCE_LENGTH);
        return -1;
    }
    printf("Pass 1 complete. Found %d possible sequences. Allocating memory...\n", total_sequences);

    // --- Memory Allocation ---
    *out_data = (float*)malloc(total_sequences * SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
    *out_labels = (int*)malloc(total_sequences * sizeof(int));
    if (!*out_data || !*out_labels) { 
        perror("Fatal: Failed to allocate memory for data"); 
        return -1; 
    }

    // --- Second Pass: Read data into memory ---
    printf("Starting pass 2: Loading data...\n");
    int current_sequence_idx = 0;
    for (int i = 0; i < num_gestures; i++) {
        snprintf(path, sizeof(path), "%s/%s", dir_path, gestures[i]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            if (strstr(dir->d_name, ".csv") == NULL) continue;

            char file_path[2048];
            snprintf(file_path, sizeof(file_path), "%s/%s", path, dir->d_name);
            FILE* file = fopen(file_path, "r");
            if (!file) continue;

            // Count lines to determine exact memory needed for this file
            int frame_count = 0;
            char line[4096];
            while (fgets(line, sizeof(line), file)) {
                frame_count++;
            }
            rewind(file); // Go back to the start of the file

            if (frame_count == 0) { fclose(file); continue; }

            // Dynamically allocate a buffer of the exact size needed
            float* temp_frames = (float*)malloc(frame_count * INPUT_SIZE * sizeof(float));
            if (temp_frames == NULL) { 
                perror("Failed to allocate temp buffer");
                fclose(file); 
                continue; 
            }

            // Read all frames into the correctly sized buffer
            int current_frame = 0;
            while (fgets(line, sizeof(line), file) && current_frame < frame_count) {
                char* token = strtok(line, ",");
                for (int feat = 0; feat < INPUT_SIZE && token != NULL; feat++) {
                    temp_frames[current_frame * INPUT_SIZE + feat] = atof(token);
                    token = strtok(NULL, ",");
                }
                current_frame++;
            }
            fclose(file);

            // Create overlapping windows from the buffer
            if (frame_count >= SEQUENCE_LENGTH) {
                for (int start_frame = 0; start_frame <= frame_count - SEQUENCE_LENGTH; start_frame += WINDOW_STRIDE) {
                    if (current_sequence_idx >= total_sequences) {
                        fprintf(stderr, "Error: About to write out of bounds! Check counting logic.\n");
                        break;
                    }
                    float* data_ptr = &((*out_data)[current_sequence_idx * SEQUENCE_LENGTH * INPUT_SIZE]);
                    memcpy(data_ptr, &temp_frames[start_frame * INPUT_SIZE], SEQUENCE_LENGTH * INPUT_SIZE * sizeof(float));
                    (*out_labels)[current_sequence_idx] = i;
                    current_sequence_idx++;
                }
            }
            free(temp_frames);
        }
        closedir(d);
    }

    *out_num_sequences = current_sequence_idx;
    printf("Pass 2 complete. Successfully loaded and augmented data into %d sequences.\n", *out_num_sequences);


    return 0; // Success
}

// --- Data Preparation ---

void shuffle_indices(int* indices, int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

void split_data(int num_sequences, float train_split, int* train_indices, int* num_train, int* val_indices, int* num_val) {
    int* all_indices = (int*)malloc(num_sequences * sizeof(int));
    for (int i = 0; i < num_sequences; ++i) {
        all_indices[i] = i;
    }

    shuffle_indices(all_indices, num_sequences);

    *num_train = (int)(num_sequences * train_split);
    *num_val = num_sequences - *num_train;

    memcpy(train_indices, all_indices, *num_train * sizeof(int));
    memcpy(val_indices, &all_indices[*num_train], *num_val * sizeof(int));

    free(all_indices);
}
