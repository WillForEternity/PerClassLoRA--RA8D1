#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "training_logic.h"

// Constants
#define DATA_DIR "../models/data"
const char* GESTURES[] = {"wave", "swipe_left", "swipe_right"};
const int NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);

// Training Hyperparameters
#define NUM_EPOCHS 500
#define LEARNING_RATE 0.001f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define TRAIN_SPLIT 0.8f

// Loss and accuracy helpers
float calculate_loss(const float* predictions, int target_label) {
    float predicted_prob = predictions[target_label];
    if (predicted_prob < 1e-9) predicted_prob = 1e-9;
    return -logf(predicted_prob);
}

float calculate_accuracy(const float* predictions, int target_label) {
    int max_index = 0;
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (predictions[i] > predictions[max_index]) {
            max_index = i;
        }
    }
    return (max_index == target_label) ? 1.0f : 0.0f;
}

int main() {
    printf("--- C Training Executable Started ---\n");
    printf("C-Based Model Training\n");

    // Load Data
    float* all_data = NULL;
    int* all_labels = NULL;
    int num_sequences = 0;
    if (load_temporal_data(DATA_DIR, GESTURES, NUM_GESTURES, &all_data, &all_labels, &num_sequences) != 0) {
        fprintf(stderr, "Failed to load data. Exiting.\n"); return 1;
    }
    printf("Loaded %d total sequences.\n", num_sequences);

    // Split data
    int num_train = 0, num_val = 0;
    int* train_indices = (int*)malloc(num_sequences * sizeof(int));
    int* val_indices = (int*)malloc(num_sequences * sizeof(int));
    split_data(num_sequences, TRAIN_SPLIT, train_indices, &num_train, val_indices, &num_val);
    printf("Split data into %d training and %d validation samples.\n", num_train, num_val);

    // Init model
    Model model;
    init_model(&model);
    
    // Diagnostic: Initial output layer weights
    printf("[TRAINING DIAGNOSTIC] Output layer weights after initialization:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  weight[%d]: %.6f\n", i, model.output_layer.weights[i]);
    }
    printf("  bias[0]: %.6f\n", model.output_layer.biases[0]);

    // Training Loop
    printf("\nStarting Training\n");
    printf("Hyperparameters: Epochs=%d, LR=%.4f, Train/Val Split=%.0f/%.0f\n", NUM_EPOCHS, LEARNING_RATE, TRAIN_SPLIT*100, (1-TRAIN_SPLIT)*100); 
    fflush(stdout);

    int timestep = 0;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        
        // Training Phase
        float total_train_loss = 0.0f;
        for (int i = 0; i < num_train; ++i) {
            timestep++;
            int sample_idx = train_indices[i];
            float* input_sequence = &all_data[sample_idx * SEQUENCE_LENGTH * INPUT_SIZE];
            int target_label = all_labels[sample_idx];

            // Forward pass to calculate outputs and loss
            forward_pass(&model, input_sequence, epoch, i);
            total_train_loss += calculate_loss(model.output_layer.output, target_label);

            // Backward pass to calculate gradients and update weights
            backward_pass(&model, input_sequence, &target_label, 1, epoch, i);
            update_weights(&model, LEARNING_RATE, BETA1, BETA2, EPSILON, timestep);
        }

        // Validation Phase
        float total_val_loss = 0.0f;
        float total_val_acc = 0.0f;
        for (int i = 0; i < num_val; ++i) {
            int sample_idx = val_indices[i];
            float* input_sequence = &all_data[sample_idx * SEQUENCE_LENGTH * INPUT_SIZE];
            int target_label = all_labels[sample_idx];

            forward_pass(&model, input_sequence, epoch, i);
            total_val_loss += calculate_loss(model.output_layer.output, target_label);
            total_val_acc += calculate_accuracy(model.output_layer.output, target_label);
        }

        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %4d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%\n", 
                   epoch + 1, NUM_EPOCHS, 
                   total_train_loss / num_train, 
                   total_val_loss / num_val, 
                   (total_val_acc / num_val) * 100.0f);
            fflush(stdout);
        }
    }

    printf("\nTraining Complete\n");
    
    // Diagnostic: Final output layer weights
    printf("[TRAINING DIAGNOSTIC] Output layer weights after training:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  weight[%d]: %.6f\n", i, model.output_layer.weights[i]);
    }
    printf("  bias[0]: %.6f\n", model.output_layer.biases[0]);

    // Save model
    printf("\n[TRAINING] Saving model to ../models/c_model.bin...\n");
    fflush(stdout);
    save_model(&model, "../models/c_model.bin");
    printf("[TRAINING] Model saved successfully.\n");
    fflush(stdout);

    // Cleanup
    printf("\nCleaning up resources...\n");
    free(all_data);
    free(all_labels);
    free(train_indices);
    free(val_indices);

    printf("Training finished.\n");
    printf("--- C Training Executable Finished ---\n");
    return 0;
}
