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

// Globals
InferenceModel g_model;
int g_model_loaded = 0; // Flag to check if model is loaded
float g_hand_landmark_data[SEQUENCE_LENGTH * INPUT_SIZE]; // Inference data buffer
const char* g_gesture_labels[NUM_CLASSES] = {"wave", "swipe_left", "swipe_right"};

// Main
int main() {
    printf("Initializing C model...\n");

    const char* model_path = "../models/c_model.bin";
    if (!load_inference_model(&g_model, model_path)) {
        fprintf(stderr, "[SERVER WARNING] Model file not found. Server is running without a model.\n");
        g_model_loaded = 0;
    } else {
        g_model_loaded = 1;
        // Diagnostic: Print loaded model weights
        printf("[DIAGNOSTIC] Model loaded successfully. Sample weights:\n");
        printf("[DIAGNOSTIC] TCN weight[0]: %.6f\n", g_model.tcn_block.weights[0]);
        printf("[DIAGNOSTIC] TCN bias[0]: %.6f\n", g_model.tcn_block.biases[0]);
        printf("[DIAGNOSTIC] Output weight[0]: %.6f\n", g_model.output_layer.weights[0]);
        printf("[DIAGNOSTIC] Output bias[0]: %.6f\n", g_model.output_layer.biases[0]);
    }

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    printf("RA8D1 C-Model Sim: Starting socket server...\n");

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { perror("socket failed"); exit(EXIT_FAILURE); }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) { perror("setsockopt"); exit(EXIT_FAILURE); }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SERVER_PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) { perror("bind failed"); exit(EXIT_FAILURE); }
    if (listen(server_fd, 3) < 0) { perror("listen"); exit(EXIT_FAILURE); }
    
    printf("Server listening on port %d\n", SERVER_PORT);
    while(1) { // Accept new connections
        printf("[SERVER] Waiting for client connection...\n");
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) { 
            perror("accept"); 
            continue; // Wait for next connection
        }
        printf("[SERVER] Client connected on socket %d. Handling persistently.\n", new_socket);

        // Handle single client connection
        while(1) {
            char send_buffer[64];
            uint32_t msg_len_net;
            ssize_t header_read = read(new_socket, &msg_len_net, sizeof(uint32_t));

            if (header_read <= 0) {
                if (header_read < 0) perror("[SERVER] Read header failed");
                printf("[SERVER] Client disconnected.\n");
                break; // Client disconnected, close socket
            }

            uint32_t msg_len = ntohl(msg_len_net);
            size_t expected_len = sizeof(g_hand_landmark_data);
            if (msg_len != expected_len) {
                fprintf(stderr, "[SERVER] Invalid message length: %u, expected %zu\n", msg_len, expected_len);
                continue; // Wait for next message
            }

            // Read data into temp buffer
            char temp_buffer[sizeof(g_hand_landmark_data)];
            ssize_t total_read = 0;
            while (total_read < msg_len) {
                ssize_t payload_read = read(new_socket, temp_buffer + total_read, msg_len - total_read);
                if (payload_read <= 0) {
                    if (payload_read < 0) perror("[SERVER] Read payload failed");
                    printf("[SERVER] Client disconnected during payload read.\n");
                    goto close_client_socket; // Cleanup and close socket
                }
                total_read += payload_read;
            }
            
            // Network to host byte order
            uint32_t* temp_uint32 = (uint32_t*)temp_buffer;
            for (int i = 0; i < (SEQUENCE_LENGTH * INPUT_SIZE); i++) {
                uint32_t net_val = temp_uint32[i];
                uint32_t host_val = ntohl(net_val);
                g_hand_landmark_data[i] = *(float*)&host_val;
            }

            // Diagnostic: Print received data
            printf("[DIAGNOSTIC] Received data. First 10 values: ");
            for (int i = 0; i < 10; ++i) {
                printf("%.3f ", g_hand_landmark_data[i]);
            }
            printf("\n");
            
            // Run inference only if model is loaded
            float prediction_output[NUM_CLASSES] = {0};
            if (!g_model_loaded) {
                // Model not loaded, send back an error/special value
                // We can use a special prediction index like -1 to signify this.
                snprintf(send_buffer, sizeof(send_buffer), "-1,0.0");
                write(new_socket, send_buffer, strlen(send_buffer));
                printf("[SERVER] Sent 'no model' response to client.\n");
                continue; // Skip the rest of the loop
            }

            forward_pass_inference(&g_model, (float*)g_hand_landmark_data, prediction_output);

            // Diagnostic: Print raw output
            printf("[DIAGNOSTIC] Raw inference output: ");
            for (int i = 0; i < NUM_CLASSES; ++i) {
                printf("class_%d=%.6f ", i, prediction_output[i]);
            }
            printf("\n");

            // Find best prediction
            int prediction = 0;
            float confidence = 0.0f;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (prediction_output[i] > confidence) {
                    confidence = prediction_output[i];
                    prediction = i;
                }
            }

            printf("[DIAGNOSTIC] Final prediction: class_%d with confidence %.6f\n", prediction, confidence);

            // Send Response 
            snprintf(send_buffer, sizeof(send_buffer), "%d,%.4f", prediction, confidence);
            write(new_socket, send_buffer, strlen(send_buffer));
        }

    close_client_socket:
        printf("[SERVER] Closing client socket %d.\n", new_socket);
        close(new_socket);
    }
    close(server_fd);
    // No free_model needed with static allocation.
    printf("C-Model server shutdown.\n");

    return 0;
}
