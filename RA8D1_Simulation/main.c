#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "mcu_constraints.h"

#define SERVER_PORT 65432
#define SERVER_IP "127.0.0.1"
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
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[RECV_BUFFER_SIZE] = {0};

    printf("RA8D1 Simulation: Socket Client Starting...\n");

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    printf("Connected to Hand Tracker server.\n");

    while (1) {
        int bytes_received = read(sock, buffer, RECV_BUFFER_SIZE - 1);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0'; // Null-terminate the received data
            parse_data(buffer);

            printf("Received %d floats: [", INPUT_BUFFER_SIZE);
            for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
                printf("%.4f%s", g_hand_landmark_data[i], (i == INPUT_BUFFER_SIZE - 1) ? "" : ", ");
            }
            printf("]\n");

        } else if (bytes_received == 0) {
            printf("Server closed connection.\n");
            break;
        } else {
            perror("read");
            break;
        }
    }

    close(sock);
    return 0;
}
