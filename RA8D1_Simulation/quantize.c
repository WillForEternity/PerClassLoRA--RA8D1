#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // For fmaxf, fminf
#include <sys/stat.h> // For stat() to get file size
#include "training_logic.h"

// Performs simple symmetric quantization on the model weights and biases.
void quantize_model(const Model* float_model, QuantizedModel* quantized_model) {
    // Quantize TCN block weights and biases
    for (size_t i = 0; i < sizeof(float_model->tcn_block.weights) / sizeof(float); ++i) {
        quantized_model->tcn_block_weights[i] = (int8_t)(fmaxf(-128.0f, fminf(127.0f, float_model->tcn_block.weights[i] * 127.0f)));
    }
    for (size_t i = 0; i < sizeof(float_model->tcn_block.biases) / sizeof(float); ++i) {
        quantized_model->tcn_block_biases[i] = (int8_t)(fmaxf(-128.0f, fminf(127.0f, float_model->tcn_block.biases[i] * 127.0f)));
    }

    // Quantize output layer weights and biases
    for (size_t i = 0; i < sizeof(float_model->output_layer.weights) / sizeof(float); ++i) {
        quantized_model->output_layer_weights[i] = (int8_t)(fmaxf(-128.0f, fminf(127.0f, float_model->output_layer.weights[i] * 127.0f)));
    }
    for (size_t i = 0; i < sizeof(float_model->output_layer.biases) / sizeof(float); ++i) {
        quantized_model->output_layer_biases[i] = (int8_t)(fmaxf(-128.0f, fminf(127.0f, float_model->output_layer.biases[i] * 127.0f)));
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_model_path> <output_model_path>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    InferenceModel temp_inference_model;
    if (!load_inference_model(&temp_inference_model, input_path)) {
        fprintf(stderr, "Error: Failed to load model from %s\n", input_path);
        return 1;
    }

    // The quantize_model function expects a `Model` struct, but we load an `InferenceModel`.
    // We'll create a dummy `Model` and copy the weights over.
    Model float_model;
    memcpy(float_model.tcn_block.weights, temp_inference_model.tcn_block.weights, sizeof(temp_inference_model.tcn_block.weights));
    memcpy(float_model.tcn_block.biases, temp_inference_model.tcn_block.biases, sizeof(temp_inference_model.tcn_block.biases));
    memcpy(float_model.output_layer.weights, temp_inference_model.output_layer.weights, sizeof(temp_inference_model.output_layer.weights));
    memcpy(float_model.output_layer.biases, temp_inference_model.output_layer.biases, sizeof(temp_inference_model.output_layer.biases));

    QuantizedModel quantized_model;
    quantize_model(&float_model, &quantized_model);

    save_quantized_model(&quantized_model, output_path);

    printf("Model quantized successfully from '%s' to '%s'\n", input_path, output_path);

    // Get and print the file size
    struct stat st;
    if (stat(output_path, &st) == 0) {
        long file_size = st.st_size;
        printf("Quantized model size: %ld bytes (%.2f KB)\n", file_size, (double)file_size / 1024.0);
    } else {
        perror("Failed to get file size");
    }

    return 0;
}
