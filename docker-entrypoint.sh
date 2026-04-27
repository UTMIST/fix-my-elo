#!/bin/sh
set -eu

if [ -n "${TEAM2_MODEL_URL:-}" ]; then
  mkdir -p "$(dirname "$TEAM2_MODEL_PATH")"

  if [ ! -s "$TEAM2_MODEL_PATH" ]; then
    echo "Downloading Team2 model from ${TEAM2_MODEL_URL} to ${TEAM2_MODEL_PATH}"
    
    # Download with validation
    if ! curl -L --fail --retry 3 --retry-delay 2 -o "$TEAM2_MODEL_PATH" "$TEAM2_MODEL_URL"; then
      echo "ERROR: Failed to download model from ${TEAM2_MODEL_URL}"
      exit 1
    fi
    
    # Validate file is not empty and looks like a PyTorch file
    # stat command differs between Linux and macOS
    FILE_SIZE=$(stat -c%s "$TEAM2_MODEL_PATH" 2>/dev/null || stat -f%z "$TEAM2_MODEL_PATH" 2>/dev/null || echo 0)
    
    if [ "$FILE_SIZE" -lt 1000 ]; then
      echo "ERROR: Downloaded file too small (${FILE_SIZE} bytes). May be an error page."
      cat "$TEAM2_MODEL_PATH"
      rm -f "$TEAM2_MODEL_PATH"
      exit 1
    fi
    
    # Check for common error signatures in downloaded file (HTML error pages)
    FILE_HEAD=$(head -c 500 "$TEAM2_MODEL_PATH" 2>/dev/null || echo "")
    case "$FILE_HEAD" in
      *"<!DOCTYPE"*|*"<html"*|*"404"*|*"error"*) 
        echo "ERROR: Downloaded file appears to be HTML (likely an error page, not a PyTorch checkpoint)"
        echo "First 500 bytes:"
        head -c 500 "$TEAM2_MODEL_PATH"
        rm -f "$TEAM2_MODEL_PATH"
        exit 1
        ;;
    esac
    
    echo "Model downloaded successfully (${FILE_SIZE} bytes)"
  else
    echo "Team2 model already present at ${TEAM2_MODEL_PATH}"
  fi
fi

# Ensure Python dependencies are present in the venv at runtime (fallback)
if [ -x "/venv/bin/python" ]; then
  echo "Checking python-chess availability in venv..."
  if ! /venv/bin/python -c "import importlib; sys_spec = importlib.util.find_spec('chess'); exit(0 if sys_spec is not None else 1)" 2>/dev/null; then
    echo "python-chess not found in venv; installing Team2 requirements into venv..."
    if [ -f "/app/Team2/requirements.txt" ]; then
      /venv/bin/pip install --no-cache-dir -r /app/Team2/requirements.txt || {
        echo "ERROR: Runtime pip install failed";
        /venv/bin/pip --version || true;
      }
    else
      echo "WARNING: /app/Team2/requirements.txt not found; cannot install runtime dependencies"
    fi
  else
    echo "python-chess available in venv"
  fi
fi

exec "$@"