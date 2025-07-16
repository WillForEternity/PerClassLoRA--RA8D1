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
#define INPUT_BUFFER_SIZE (INPUT_SIZE) // Use macro from our header
#define RECV_BUFFER_SIZE 1024

// --- Global Variables ---
static float g_hand_landmark_data[INPUT_BUFFER_SIZE];
static Model g_model;
const char* g_gesture_labels[NUM_CLASSES] = {"fist", "palm", "pointing"};

// --- Helper Functions ---

void parse_data(char* buffer) {
    char* token = strtok(buffer, ",");
    int i = 0;
    while (token != NULL && i < INPUT_BUFFER_SIZE) {
        g_hand_landmark_data[i++] = atof(token);
        token = strtok(NULL, ",");
    }
}

void run_inference(int client_socket) {

    // Use our forward_pass function
    forward_pass(&g_model, g_hand_landmark_data, 0, 0); // epoch and sample_idx are not used in inference

    // Get the output from our model's output layer
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
    while(1) {
        printf("Waiting for a client connection...\n");
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) { 
            perror("accept"); 
            continue;
        }
        printf("Client connected. Waiting for data...\n");

        while (1) {
            ssize_t bytes_received = read(new_socket, buffer, RECV_BUFFER_SIZE - 1);
            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';
                parse_data(buffer);
                run_inference(new_socket);
            } else {
                printf("Client disconnected.\n");
                close(new_socket);
                break;
            }
        }
    }
    close(server_fd);
    // free_model is no longer needed with static allocation.
    printf("C-Model server shutting down.\n");

    return 0;
}
