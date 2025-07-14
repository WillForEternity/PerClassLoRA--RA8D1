#!/bin/bash

# --- Configuration ---
PROJECT_ROOT=$(pwd)
TRACKER_VENV="$PROJECT_ROOT/Python_Hand_Tracker/venv_tracker"
TRAINING_VENV="$PROJECT_ROOT/Python_Hand_Tracker/venv_training"
SIM_DIR="$PROJECT_ROOT/RA8D1_Simulation"
SIM_EXEC="./simulation"

# --- Helper Functions ---

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO] $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS] $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m[ERROR] $1\033[0m"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to clean up background processes
cleanup() {
    print_info "Cleaning up background processes..."
    if [ -n "$sim_pid" ]; then
        kill "$sim_pid" 2>/dev/null
        print_success "C simulation server stopped."
    fi
    exit 0
}

# Trap EXIT signal to run cleanup function
trap cleanup EXIT

# --- Main Logic ---

# Check for correct argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {collect|train|inference|reinstall}"
    exit 1
fi

COMMAND=$1

if [ "$COMMAND" = "setup" ]; then
    print_info "Setting up the project..."
    # Create Python virtual environment for training
    print_info "Creating Python virtual environment for training with Python 3.11..."
    if ! command_exists python3.11; then
        print_error "Python 3.11 not found. Please install Python 3.11 and ensure it's in your PATH."
    fi
    python3.11 -m venv "$TRAINING_VENV"
    print_success "Training virtual environment created."

    # Install Python dependencies
    print_info "Installing Python dependencies..."
    source "$TRAINING_VENV/bin/activate"
    pip install -r "$PROJECT_ROOT/Python_Hand_Tracker/requirements_training.txt"
    print_success "Python dependencies for training installed."

    # Create and set up the tracker virtual environment
    print_info "Creating Python virtual environment for tracking with Python 3.11..."
    if ! command_exists python3.11; then
        print_error "Python 3.11 not found. Please install Python 3.11 and ensure it's in your PATH."
    fi
    python3.11 -m venv "$TRACKER_VENV"
    print_success "Tracker virtual environment created."

    print_info "Installing Python dependencies for tracking..."
    source "$TRACKER_VENV/bin/activate"
    pip install -r "$PROJECT_ROOT/Python_Hand_Tracker/requirements_tracker.txt"
    print_success "Python dependencies for tracking installed."

    # Build C simulation
    print_info "Building C simulation..."
    (cd "$SIM_DIR" && make clean && make)
    print_success "C simulation built."
    print_success "Setup complete."
    exit 0

elif [ "$COMMAND" = "reinstall" ]; then
    print_info "Reinstalling Python dependencies for the training environment..."
    source "$TRAINING_VENV/bin/activate"
    pip install -r "$PROJECT_ROOT/Python_Hand_Tracker/requirements_training.txt"
    print_success "Dependencies reinstalled."
    exit 0
fi

MODE=$1

case $MODE in
    collect)
        print_info "Starting data collection mode..."
        if [ ! -d "$TRACKER_VENV" ]; then
            print_error "Tracker virtual environment not found. Please run the setup first."
        fi
        source "$TRACKER_VENV/bin/activate"
        python "$PROJECT_ROOT/Python_Hand_Tracker/hand_tracker.py" --mode collect
        print_success "Data collection finished."
        ;;

    train)
        print_info "Starting model training..."
        if [ ! -f "$TRAINING_VENV/bin/activate" ]; then
            print_error "Training virtual environment not found. Please run './run.sh setup' first."
            exit 1
        fi
        source "$TRAINING_VENV/bin/activate"
        python "$PROJECT_ROOT/Python_Hand_Tracker/train_model.py"
        print_success "Model training finished."
        ;;

    inference)
        print_info "Starting end-to-end inference simulation..."
        
        # 1. Compile C simulation if necessary
        if [ ! -f "$SIM_EXEC" ]; then
            print_info "C simulation executable not found. Compiling..."
            if ! command_exists gcc; then
                print_error "'gcc' command not found. Please install GCC."
            fi
            (cd "$SIM_DIR" && gcc main.c -o simulation -I/opt/homebrew/opt/onnxruntime/include -L/opt/homebrew/opt/onnxruntime/lib -lonnxruntime) || print_error "C compilation failed."
            print_success "C simulation compiled successfully."
        fi

        # 2. Start C simulation server in the background
        # Define a cleanup function to be called on exit
        cleanup() {
            print_info "Cleaning up background processes..."
            if ps -p $SIM_PID > /dev/null; then
                kill $SIM_PID
            fi
            # The pkill is a fallback for any stubborn processes
            pkill -f "$SIM_EXEC"
            print_success "C simulation server stopped."
        }
        trap cleanup EXIT

        # Force kill any lingering simulation processes to free up the port
        pkill -f "$SIM_EXEC"

        print_info "Starting C simulation server in the background..."
        (cd "$SIM_DIR" && $SIM_EXEC) &
        SIM_PID=$!
        sleep 1 # Give the server a moment to start

        if [ ! -f "$PROJECT_ROOT/models/model.onnx" ]; then
            print_error "Model file not found. Please run './run.sh train' first."
        fi

        print_info "Starting Python hand tracker client..."
        source "$TRACKER_VENV/bin/activate"
        # Run python and wait for it to complete
        python "$PROJECT_ROOT/Python_Hand_Tracker/hand_tracker.py" --mode inference
        wait $SIM_PID
        print_success "Inference session finished."
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: ./run.sh {setup|collect|train|inference}"
        exit 1
        ;;
esac

exit 0
