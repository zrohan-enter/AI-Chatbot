#!/bin/bash

# This script assumes you are running it from the "D:/MY Ai/Indently Pro VS" directory.
# First, activate the virtual environment.
source .venv_official/Scripts/activate

# Then, run the main.py script using the Python executable from the activated virtual environment.
# This ensures that all dependencies installed in your venv are available.
./.venv_official/Scripts/python.exe main.py

# Optional: Deactivate the virtual environment after the script finishes.
# You can comment out the 'deactivate' line if you prefer to keep the environment active
# in your terminal session after the chatbot exits.
deactivate