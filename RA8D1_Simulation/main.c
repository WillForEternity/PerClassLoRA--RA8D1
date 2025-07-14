#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "mcu_constraints.h"

#define SERVER_PORT 65432
#define NUM_LANDMARKS 6
#define NUM_COORDS 3
#define INPUT_BUFFER_SIZE (NUM_LANDMARKS * NUM_COORDS)
#define RECV_BUFFER_SIZE 1024

// Statically allocated buffer for hand landmark data, as if on the MCU
static float g_hand_landmark_data[INPUT_BUFFER_SIZE];

void parse_data(char* buffer) {
    char* token = strtok(buffer, ",");
    int i = 0;
    while (token != NULL && i < INPUT_BUFFER_SIZE) {
        g_hand_landmark_data[i++] = atof(token);
        token = strtok(NULL, ",");
    }
}

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[RECV_BUFFER_SIZE] = {0};

    printf("RA8D1 Simulation: Socket Server Starting...\n");

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SERVER_PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    printf("Server listening on port %d\n", SERVER_PORT);
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    printf("Client connected.\n");

    while (1) {
        int bytes_received = read(new_socket, buffer, RECV_BUFFER_SIZE - 1);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            parse_data(buffer);

            printf("Received %d floats: [", INPUT_BUFFER_SIZE);
            for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
                printf("%.4f%s", g_hand_landmark_data[i], (i == INPUT_BUFFER_SIZE - 1) ? "" : ", ");
            }
            printf("]\n");

        } else if (bytes_received == 0) {
            printf("Client disconnected.\n");
            break;
        } else {
            perror("read");
            break;
        }
    }

    close(new_socket);
    close(server_fd);
    return 0;
}
