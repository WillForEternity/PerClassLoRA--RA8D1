CC=gcc
# Common flags, -g adds debug symbols
CFLAGS=-Wall -g -I.

# --- Target-specific variables ---
SIM_TARGET=ra8d1_sim
TRAIN_TARGET=train_c

# Linker flags for the simulator (needs ONNX Runtime)
SIM_LDFLAGS=-L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime
# Linker flags for the trainer (needs math library for functions like exp, sqrt)
TRAIN_LDFLAGS=-lm

# --- Build Rules ---

# Default target: build both applications
all: $(SIM_TARGET) $(TRAIN_TARGET)

# Rule to build the inference simulator
$(SIM_TARGET): main.c training_logic.c
	$(CC) $(CFLAGS) -o $@ $^ $(SIM_LDFLAGS)

# Rule to build the C-based trainer
$(TRAIN_TARGET): train_in_c.c training_logic.c
	$(CC) $(CFLAGS) -o $@ $^ $(TRAIN_LDFLAGS)

# Rule to clean build artifacts
clean:
	rm -f $(SIM_TARGET) $(TRAIN_TARGET)
