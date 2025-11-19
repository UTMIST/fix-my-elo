#!/usr/bin/env zsh
# Run using 'source scripts/venv.sh'

VENV_DIR="${1:-.venv}"
PY_CMD="${2:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv: $VENV_DIR"
  $PY_CMD -m venv "$VENV_DIR" || { echo "Failed to create venv."; return 1; }

  # Install requirements
  [ -f requirements.txt ] && "$VENV_DIR/bin/pip" install -r requirements.txt
fi

# Activate venv
echo "Activating $VENV_DIR..."
source "$VENV_DIR/bin/activate" || { echo "Activation failed."; return 1; }

echo "Activated."
