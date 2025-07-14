import subprocess
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRACKER_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_tracker")
TRAINING_VENV = os.path.join(PROJECT_ROOT, "Python_Hand_Tracker", "venv_training")
SIM_DIR = os.path.join(PROJECT_ROOT, "RA8D1_Simulation")

def run_command(command, cwd=PROJECT_ROOT):
    """Runs a command and yields its output line by line."""
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        shell=True, 
        cwd=cwd
    )
    for line in iter(process.stdout.readline, ''):
        yield line
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def run_setup():
    """Runs the full project setup process."""
    try:
        yield "--- Starting Project Setup ---\n"
        
        # Check for Python 3.11
        yield "[INFO] Checking for Python 3.11...\n"
        if "3.11" not in sys.version:
             yield f"[WARNING] You are running Python {sys.version}. The project uses 3.11. Proceeding, but errors may occur.\n"
        else:
            yield "[SUCCESS] Python 3.11 found.\n"

        # Create Training Venv
        yield "\n[INFO] Creating Python virtual environment for training...\n"
        yield from run_command(f'python3.11 -m venv "{TRAINING_VENV}"')
        yield "[SUCCESS] Training virtual environment created.\n"

        # Install Training Dependencies
        yield "\n[INFO] Installing training dependencies...\n"
        pip_executable = os.path.join(TRAINING_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_training.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "[SUCCESS] Training dependencies installed.\n"

        # Create Tracker Venv
        yield "\n[INFO] Creating Python virtual environment for tracking...\n"
        yield from run_command(f'python3.11 -m venv "{TRACKER_VENV}"')
        yield "[SUCCESS] Tracker virtual environment created.\n"

        # Install Tracker Dependencies
        yield "\n[INFO] Installing tracking dependencies...\n"
        pip_executable = os.path.join(TRACKER_VENV, 'bin', 'pip')
        req_file = os.path.join(PROJECT_ROOT, 'Python_Hand_Tracker', 'requirements_tracker.txt')
        yield from run_command(f'"{pip_executable}" install -r "{req_file}"')
        yield "[SUCCESS] Tracking dependencies installed.\n"

        # Build C Simulation
        yield "\n[INFO] Building C simulation...\n"
        yield from run_command('make clean && make', cwd=SIM_DIR)
        yield "[SUCCESS] C simulation built.\n"

        yield "\n--- Setup Complete! ---\n"

    except subprocess.CalledProcessError as e:
        yield f"\n[ERROR] A command failed with exit code {e.returncode}.\n"
        yield str(e)
    except Exception as e:
        yield f"\n[ERROR] An unexpected error occurred: {str(e)}\n"
