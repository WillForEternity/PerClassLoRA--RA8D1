#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "training_logic.h"

// --- Constants ---
#define DATA_DIR "../models/data"
const char* GESTURES[] = {"wave", "swipe_left", "swipe_right"};
const int NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);

// --- Training Hyperparameters ---
#define NUM_EPOCHS 2000
#define LEARNING_RATE 0.01f

// --- Helper Function to calculate loss ---
float calculate_loss(const float* predictions, int target_label) {
    // Sparse Categorical Cross-Entropy Loss
    // Ensure the prediction for the target class is not zero to avoid log(0)
    float predicted_prob = predictions[target_label];
    if (predicted_prob < 1e-9) {
        predicted_prob = 1e-9;
    }
    return -logf(predicted_prob);
}

int main() {
    printf("--- C-Based Model Training ---\n"); fflush(stdout);

    // 1. Load Data
    printf("\n1. Loading Data from '%s'...\n", DATA_DIR); fflush(stdout);
    float* all_data = NULL;
    int* all_labels = NULL;
    int num_sequences = 0;

    if (load_temporal_data(DATA_DIR, GESTURES, NUM_GESTURES, &all_data, &all_labels, &num_sequences) != 0) {
        fprintf(stderr, "Failed to load temporal data. Exiting.\n");
        return 1;
    }

    printf("Loaded %d sequences.\n", num_sequences); fflush(stdout);

    // 2. Initialize Model
    printf("\n2. Initializing Neural Network Model...\n"); fflush(stdout);
    Model model;
    init_model(&model);

    // 3. Training Loop
    printf("\n3. Starting Training Loop...\n"); fflush(stdout);
    printf("Hyperparameters: Epochs=%d, Learning Rate=%.6f\n", NUM_EPOCHS, LEARNING_RATE); fflush(stdout);

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        shuffle_data(all_data, all_labels, num_sequences);
        float total_loss = 0.0f;

        for (int i = 0; i < num_sequences; ++i) {
            // Each 'sample' is now a sequence of frames
            float* input_sequence = &all_data[i * SEQUENCE_LENGTH * INPUT_SIZE];
            int* target_label = &all_labels[i];

            // --- Core Training Step ---
            // a) Forward pass to get predictions
            forward_pass(&model, input_sequence, epoch, i);

            // b) Calculate loss
            float loss = calculate_loss(model.output_layer.output, *target_label);
            total_loss += loss;

            // c) Backward pass to compute gradients
            backward_pass(&model, input_sequence, target_label, 1, epoch, i);

            // d) Update weights with the optimizer
            update_weights(&model, LEARNING_RATE);
        }

        // Print average loss for the epoch, but not for every single one.
        if ((epoch + 1) % 10 == 0) {
             printf("Epoch %d/%d, Average Loss: %f\n", epoch + 1, NUM_EPOCHS, total_loss / num_sequences); fflush(stdout);
        }
    }

    // 4. Save the trained model
    save_model(&model, "../models/c_model.bin");

    // 5. Cleaning up resources
    printf("\n5. Cleaning up resources...\n"); fflush(stdout);
    // free_model is no longer needed with static allocation.
    free(all_data); // Dataset memory is still dynamic.
    free(all_labels);


    printf("\n--- Training process finished successfully. ---\n"); fflush(stdout);

    return 0;
}
