#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "training_logic.h"
#include "mcu_constraints.h"

#define SERVER_PORT 65432
#define INPUT_BUFFER_SIZE (SEQUENCE_LENGTH * INPUT_SIZE)

// --- Global Variables ---
InferenceModel g_model;
float g_hand_landmark_data[SEQUENCE_LENGTH * INPUT_SIZE]; // Global buffer for inference data
const char* g_gesture_labels[NUM_CLASSES] = {"wave", "swipe_left", "swipe_right"};

// --- Main Execution ---
int main() {
    printf("Initializing C-based model...\n");

    const char* model_path = "../models/c_model.bin";
    if (!load_inference_model(&g_model, model_path)) {
        fprintf(stderr, "Error reading model file\n");
        // We cannot proceed without a valid model.
        return 1; 
    }
    
    // Diagnostic: Print some model weights to verify loading
    printf("[DIAGNOSTIC] Model loaded successfully. Sample weights:\n");
    printf("[DIAGNOSTIC] TCN weight[0]: %.6f\n", g_model.tcn_block.weights[0]);
    printf("[DIAGNOSTIC] TCN bias[0]: %.6f\n", g_model.tcn_block.biases[0]);
    printf("[DIAGNOSTIC] Output weight[0]: %.6f\n", g_model.output_layer.weights[0]);
    printf("[DIAGNOSTIC] Output bias[0]: %.6f\n", g_model.output_layer.biases[0]);

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    printf("RA8D1 C-Model Simulation: Socket Server Starting...\n");

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { perror("socket failed"); exit(EXIT_FAILURE); }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) { perror("setsockopt"); exit(EXIT_FAILURE); }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SERVER_PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) { perror("bind failed"); exit(EXIT_FAILURE); }
    if (listen(server_fd, 3) < 0) { perror("listen"); exit(EXIT_FAILURE); }
    
    printf("Server listening on port %d\n", SERVER_PORT);
    while(1) { // Outer loop to accept new connections
        printf("[SERVER] Waiting for a new client connection...\n");
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) { 
            perror("accept"); 
            continue; // Go back to waiting for a connection
        }
        printf("[SERVER] Client connected on socket %d. Entering persistent handling loop.\n", new_socket);

        // Persistent loop to handle a single client connection
        while(1) {
            uint32_t msg_len_net;
            ssize_t header_read = read(new_socket, &msg_len_net, sizeof(uint32_t));

            if (header_read <= 0) {
                if (header_read < 0) perror("[SERVER] Read header failed");
                printf("[SERVER] Client disconnected.\n");
                break; // Break inner loop to close socket
            }

            uint32_t msg_len = ntohl(msg_len_net);
            size_t expected_len = sizeof(g_hand_landmark_data);
            if (msg_len != expected_len) {
                fprintf(stderr, "[SERVER] Invalid message length: %u, expected %zu\n", msg_len, expected_len);
                continue; // Wait for next message
            }

            // Read data into a temporary buffer first
            char temp_buffer[sizeof(g_hand_landmark_data)];
            ssize_t total_read = 0;
            while (total_read < msg_len) {
                ssize_t payload_read = read(new_socket, temp_buffer + total_read, msg_len - total_read);
                if (payload_read <= 0) {
                    if (payload_read < 0) perror("[SERVER] Read payload failed");
                    printf("[SERVER] Client disconnected during payload read.\n");
                    goto close_client_socket; // Use goto for shared cleanup
                }
                total_read += payload_read;
            }
            
            // Convert from network byte order to host byte order
            uint32_t* temp_uint32 = (uint32_t*)temp_buffer;
            for (int i = 0; i < (SEQUENCE_LENGTH * INPUT_SIZE); i++) {
                uint32_t net_val = temp_uint32[i];
                uint32_t host_val = ntohl(net_val);
                g_hand_landmark_data[i] = *(float*)&host_val;
            }

            // Diagnostic: Print sample input data
            printf("[DIAGNOSTIC] Received data. First 10 values: ");
            for (int i = 0; i < 10; ++i) {
                printf("%.3f ", g_hand_landmark_data[i]);
            }
            printf("\n");
            
            // Run inference
            float prediction_output[NUM_CLASSES];
            forward_pass_inference(&g_model, (float*)g_hand_landmark_data, prediction_output);

            // Diagnostic: Print raw inference output
            printf("[DIAGNOSTIC] Raw inference output: ");
            for (int i = 0; i < NUM_CLASSES; ++i) {
                printf("class_%d=%.6f ", i, prediction_output[i]);
            }
            printf("\n");

            // Find the prediction with the highest confidence
            int prediction = 0;
            float confidence = 0.0f;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (prediction_output[i] > confidence) {
                    confidence = prediction_output[i];
                    prediction = i;
                }
            }

            printf("[DIAGNOSTIC] Final prediction: class_%d with confidence %.6f\n", prediction, confidence);

            // --- Send Response --- 
            char send_buffer[64];
            snprintf(send_buffer, sizeof(send_buffer), "%d,%.4f", prediction, confidence);
            write(new_socket, send_buffer, strlen(send_buffer));
        }

    close_client_socket:
        printf("[SERVER] Closing client socket %d.\n", new_socket);
        close(new_socket);
    }
    close(server_fd);
    // free_model is no longer needed with static allocation.
    printf("C-Model server shutting down.\n");

    return 0;
}
