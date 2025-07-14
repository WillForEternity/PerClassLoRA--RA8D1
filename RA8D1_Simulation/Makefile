CC=gcc
CFLAGS=-Wall -I. -I/opt/homebrew/opt/onnxruntime/include
LDFLAGS=-L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime
TARGET=ra8d1_sim

all: $(TARGET)

$(TARGET): main.c
		$(CC) $(CFLAGS) -o $(TARGET) main.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
