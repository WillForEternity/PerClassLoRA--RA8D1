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
#define RECV_BUFFER_SIZE (INPUT_BUFFER_SIZE * 12) // Generous buffer for string representation

// --- Global Variables ---
static float g_hand_landmark_data[INPUT_BUFFER_SIZE];
static Model g_model;
const char* g_gesture_labels[NUM_CLASSES] = {"wave", "swipe_left", "swipe_right"};

// --- Helper Functions ---

// Parses a complete sequence of landmark data from the buffer
void parse_data(char* buffer) {
    char* saveptr; // For strtok_r
    char* token = strtok_r(buffer, ",\n", &saveptr);
    int i = 0;
    while (token != NULL && i < INPUT_BUFFER_SIZE) {
        g_hand_landmark_data[i++] = atof(token);
        token = strtok_r(NULL, ",\n", &saveptr);
    }
}

void run_inference(int client_socket) {
    // The entire sequence is now in g_hand_landmark_data
    forward_pass(&g_model, g_hand_landmark_data, 0, 0); // epoch and sample_idx are not used in inference

    float* output_data = g_model.output_layer.output;

    int max_index = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > output_data[max_index]) {
            max_index = i;
        }
    }
    float confidence = output_data[max_index];

    // Send the result back to the client
    char response_buffer[64];
    snprintf(response_buffer, sizeof(response_buffer), "%d,%.4f\n", max_index, confidence);
    send(client_socket, response_buffer, strlen(response_buffer), 0);
}

// --- Main Execution ---
int main() {
    printf("Initializing C-based model...\n");
    // init_model(&g_model); // No longer needed for static model, weights are loaded or random

    const char* model_path = "../models/c_model.bin";
    if (load_model(&g_model, model_path) != 0) {
        fprintf(stderr, "Failed to load C model. Using random weights. Consider training first.\n");
        // If model fails to load, we can proceed with a randomly initialized model for testing.
        init_model(&g_model);
    }

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[RECV_BUFFER_SIZE] = {0};

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

        // Inner loop to handle one client persistently
        while(1) {
            ssize_t bytes_received = read(new_socket, buffer, RECV_BUFFER_SIZE - 1);
            
            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';
                // No need to print data received every time, it's too noisy
                // printf("[HANDLER] Received %zd bytes. Parsing and running inference...\n", bytes_received);
                parse_data(buffer);
                run_inference(new_socket);
                // printf("[HANDLER] Inference complete, response sent.\n");
            } else {
                if (bytes_received == 0) {
                    printf("[HANDLER] Client on socket %d disconnected gracefully.\n", new_socket);
                } else {
                    perror("[HANDLER] read failed");
                }
                break; // Exit inner loop to close socket
            }
        }

        printf("[SERVER] Closing socket %d and waiting for new connection.\n", new_socket);
        close(new_socket);
    }
    close(server_fd);
    // free_model is no longer needed with static allocation.
    printf("C-Model server shutting down.\n");

    return 0;
}
