#!/bin/bash

# Virtual environment name
VENV_NAME="venv"

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_NAME" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_NAME" || python -m venv "$VENV_NAME"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
  fi
fi

# Detect operating system and activate virtual environment accordingly
case "$(uname -s)" in
  Linux*|Darwin*) # Linux or macOS
    source "$VENV_NAME/bin/activate"
    PIP_COMMAND="python3 -m pip" #explicitly use python3
    ;;
  MINGW64*|MSYS_NT*) # Windows (Git Bash)
    source "$VENV_NAME/Scripts/activate"
    PIP_COMMAND="python -m pip"
    ;;
  CYGWIN_NT*) #Cygwin
    source "$VENV_NAME/Scripts/activate"
    PIP_COMMAND="python -m pip"
    ;;
  *)
    echo "Error: Operating system not recognized."
    exit 1
    ;;
esac

# Install dependencies from requirements.txt
$PIP_COMMAND install --upgrade pip
$PIP_COMMAND install -r requirements.txt

# Run the Streamlit app
streamlit run app.py