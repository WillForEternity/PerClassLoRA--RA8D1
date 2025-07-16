#!/bin/bash

# Hand Gesture Recognition App Startup Script
# This script starts the C inference server and then launches the GUI app

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_SERVER_DIR="$PROJECT_ROOT/RA8D1_Simulation"
GUI_VENV="$PROJECT_ROOT/gui_app/venv_gui"
PID_FILE="$PROJECT_ROOT/.c_server.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up processes...${NC}"
    
    # Kill the C server if it's running
    if [ -f "$PID_FILE" ]; then
        C_SERVER_PID=$(cat "$PID_FILE")
        if kill -0 "$C_SERVER_PID" 2>/dev/null; then
            echo "Stopping C inference server (PID: $C_SERVER_PID)..."
            kill "$C_SERVER_PID"
            wait "$C_SERVER_PID" 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
    
    echo -e "${GREEN}Cleanup complete.${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

echo -e "${GREEN}=== Hand Gesture Recognition App Startup ===${NC}"

# Check if C executable exists
if [ ! -f "$C_SERVER_DIR/ra8d1_sim" ]; then
    echo -e "${RED}Error: C inference server not found at $C_SERVER_DIR/ra8d1_sim${NC}"
    echo "Please run 'make' in the RA8D1_Simulation directory first."
    exit 1
fi

# Check if GUI virtual environment exists
if [ ! -d "$GUI_VENV" ]; then
    echo -e "${RED}Error: GUI virtual environment not found at $GUI_VENV${NC}"
    echo "Please run the setup process first."
    exit 1
fi

# Check if model file exists
if [ ! -f "$PROJECT_ROOT/models/c_model.bin" ]; then
    echo -e "${YELLOW}Warning: c_model.bin not found. Please train the model first.${NC}"
fi

echo -e "${YELLOW}Step 1: Starting C inference server...${NC}"

# Start the C server in the background
cd "$C_SERVER_DIR"
./ra8d1_sim &
C_SERVER_PID=$!

# Save the PID for cleanup
echo "$C_SERVER_PID" > "$PID_FILE"

echo "C inference server started (PID: $C_SERVER_PID)"

# Wait a moment for the server to initialize
echo "Waiting for C server to initialize..."
sleep 3

# Check if the server is still running
if ! kill -0 "$C_SERVER_PID" 2>/dev/null; then
    echo -e "${RED}Error: C inference server failed to start or crashed immediately.${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

# Test if the server is accepting connections
echo "Testing server connection..."
python3 -c "
import socket
import time
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(('localhost', 65432))
    sock.close()
    print('✓ C server is accepting connections')
except Exception as e:
    print(f'✗ Connection test failed: {e}')
    exit(1)
" || {
    echo -e "${RED}Error: C server is not accepting connections.${NC}"
    cleanup
    exit 1
}

echo -e "${GREEN}✓ C inference server is running and ready${NC}"

echo -e "${YELLOW}Step 2: Starting GUI application...${NC}"

# Start the GUI app
cd "$PROJECT_ROOT"
source "$GUI_VENV/bin/activate"
python gui_app/main_app.py

# The GUI app will run until the user closes it
# When it exits, the cleanup function will be called automatically
