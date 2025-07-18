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

# Dependencies
# Both targets depend on the logic source and header
$(TRAIN_TARGET): train_in_c.o $(LOGIC_SRC)
$(SIM_TARGET): main.o $(LOGIC_SRC)

# Rule to build the executables
$(TRAIN_TARGET): train_in_c.o training_logic.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SIM_TARGET): main.o training_logic.o
	$(CC) $(CFLAGS) -o $@ $^ -L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime

# Generic rule for object files
# This ensures .o files are rebuilt if the header changes
%.o: %.c $(LOGIC_HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Phony targets are not files
.PHONY: all clean

# Rule to clean build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf *.dSYM
	rm -f $(SIM_TARGET) $(TRAIN_TARGET) ra8d1_sim* train_c* simulation
