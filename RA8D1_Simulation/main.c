#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include "mcu_constraints.h"

#define SERVER_PORT 65432
#define NUM_LANDMARKS 6
#define NUM_COORDS 3
#define NUM_CLASSES 3
#define INPUT_BUFFER_SIZE (NUM_LANDMARKS * NUM_COORDS)
#define RECV_BUFFER_SIZE 1024

// --- Global Variables ---
static float g_hand_landmark_data[INPUT_BUFFER_SIZE];
const OrtApi* g_ort = NULL;
OrtEnv* g_env = NULL;
OrtSession* g_session = NULL;
const char* g_gesture_labels[NUM_CLASSES] = {"fist", "palm", "pointing"};

// --- Helper Functions ---
void CheckStatus(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

void initialize_onnx() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &g_env));
    
    OrtSessionOptions* session_options;
    CheckStatus(g_ort->CreateSessionOptions(&session_options));

    const char* model_path = "../models/model.onnx";
    CheckStatus(g_ort->CreateSession(g_env, model_path, session_options, &g_session));
    printf("ONNX Runtime initialized successfully. Model loaded from %s\n", model_path);

    g_ort->ReleaseSessionOptions(session_options);
}

void parse_data(char* buffer) {
    char* token = strtok(buffer, ",");
    int i = 0;
    while (token != NULL && i < INPUT_BUFFER_SIZE) {
        g_hand_landmark_data[i++] = atof(token);
        token = strtok(NULL, ",");
    }
}

void run_inference() {
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    int64_t input_shape[] = {1, INPUT_BUFFER_SIZE};
    OrtValue* input_tensor = NULL;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, &g_hand_landmark_data, sizeof(g_hand_landmark_data), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;

    CheckStatus(g_ort->Run(g_session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));

    float* output_data;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    int max_index = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > output_data[max_index]) {
            max_index = i;
        }
    }
    printf("Prediction: %s\n", g_gesture_labels[max_index]);

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
}

// --- Main Execution ---
int main() {
    initialize_onnx();

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[RECV_BUFFER_SIZE] = {0};

    printf("RA8D1 Simulation: Socket Server Starting...\n");

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
            continue; // Continue to the next iteration to wait for a connection
        }
        printf("Client connected. Waiting for data...\n");

        while (1) {
            ssize_t bytes_received = read(new_socket, buffer, RECV_BUFFER_SIZE - 1);
            printf("Bytes received: %zd\n", bytes_received);
            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';
                parse_data(buffer);
                printf("Running inference...\n");
                run_inference();
            } else {
                printf("Client disconnected.\n");
                close(new_socket);
                break; // Break from inner loop to wait for a new connection
            }
        }
    }
    close(server_fd);
    g_ort->ReleaseSession(g_session);
    g_ort->ReleaseEnv(g_env);
    printf("ONNX Runtime cleaned up. Exiting.\n");

    return 0;
}
