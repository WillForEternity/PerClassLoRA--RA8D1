CC=gcc
CFLAGS=-Wall -I.
TARGET=ra8d1_sim

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $(TARGET) main.c

clean:
	rm -f $(TARGET)
