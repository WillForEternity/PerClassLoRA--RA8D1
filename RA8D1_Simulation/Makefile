CC=gcc
CFLAGS=-Wall -g -I.
LDFLAGS_SIM=-L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime
LDFLAGS_TRAIN=-lm

# --- Targets ---
SIM_TARGET=ra8d1_sim
TRAIN_TARGET=train_c
QUANTIZE_TARGET=quantize

# --- Source & Object Files ---
SIM_SRCS=main.c training_logic.c
TRAIN_SRCS=train_in_c.c training_logic.c
QUANTIZE_SRCS=quantize.c training_logic.c

SIM_OBJS=$(SIM_SRCS:.c=.o)
TRAIN_OBJS=$(TRAIN_SRCS:.c=.o)
QUANTIZE_OBJS=$(QUANTIZE_SRCS:.c=.o)

# --- Build Rules ---
all: $(SIM_TARGET) $(TRAIN_TARGET) $(QUANTIZE_TARGET)

$(SIM_TARGET): $(SIM_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_SIM)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_TRAIN)

$(QUANTIZE_TARGET): $(QUANTIZE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_TRAIN)

# Generic rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# --- Housekeeping ---
.PHONY: all clean

clean:
	@echo "Cleaning up build artifacts..."
	rm -f $(SIM_TARGET) $(TRAIN_TARGET) $(QUANTIZE_TARGET) *.o *.dSYM
