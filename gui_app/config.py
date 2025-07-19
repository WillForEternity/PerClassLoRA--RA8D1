import json
import os

# Define the path to the configuration file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'gui_app', 'gestures.json')

DEFAULT_GESTURES = ["wave", "swipe_left", "swipe_right"]

def load_gestures():
    """Loads the list of gesture names from the config file."""
    if not os.path.exists(CONFIG_FILE):
        # Create the file with default values if it doesn't exist
        save_gestures(DEFAULT_GESTURES)
        return DEFAULT_GESTURES
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            # Basic validation to ensure it's a list of strings
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return data
            else:
                # If data is corrupt, fall back to defaults
                save_gestures(DEFAULT_GESTURES)
                return DEFAULT_GESTURES
    except (json.JSONDecodeError, IOError):
        # If file is empty, corrupt, or unreadable, fall back to defaults
        save_gestures(DEFAULT_GESTURES)
        return DEFAULT_GESTURES

def save_gestures(gestures):
    """Saves the list of gesture names to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(gestures, f, indent=4)
