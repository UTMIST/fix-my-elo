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
PYTHON_BIN="${TEAM2_PYTHON:-}"
if [ -x "/venv/bin/python" ]; then
  # Prefer the venv python even if TEAM2_PYTHON is set to system python.
  PYTHON_BIN="/venv/bin/python"
elif [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="/usr/bin/python3"
fi

if [ -x "$PYTHON_BIN" ]; then
  echo "Using Python executable: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "Checking python-chess availability using $PYTHON_BIN..."
  if ! $PYTHON_BIN - <<'PY'
import sys
try:
    import chess
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  then
    echo "python-chess not found in venv; installing Team2 requirements into venv..."
    if [ -f "/app/Team2/requirements.txt" ]; then
      "$PYTHON_BIN" -m pip install --no-cache-dir -r /app/Team2/requirements.txt || {
        echo "ERROR: Runtime pip install failed";
        "$PYTHON_BIN" -m pip --version || true;
      }
    else
      echo "WARNING: /app/Team2/requirements.txt not found; cannot install runtime dependencies"
    fi
  else
    echo "python-chess available in venv"
  fi

  # Print a quick package summary for debugging
  echo "--- pip list (top) ---"
  "$PYTHON_BIN" -m pip show chess 2>/dev/null || true
  "$PYTHON_BIN" -m pip show torch 2>/dev/null || true
  echo "----------------------"
fi

# Start the persistent engine server (loads model once)
ENGINE_ENABLE="${TEAM2_ENGINE_ENABLE:-1}"
ENGINE_PORT="${TEAM2_ENGINE_PORT:-8001}"
ENGINE_HOST="${TEAM2_ENGINE_HOST:-127.0.0.1}"
if [ "$ENGINE_ENABLE" != "0" ]; then
  if [ -x "$PYTHON_BIN" ]; then
    echo "Starting Team2 engine server on ${ENGINE_HOST}:${ENGINE_PORT}"
    TEAM2_ENGINE_HOST="$ENGINE_HOST" TEAM2_ENGINE_PORT="$ENGINE_PORT" "$PYTHON_BIN" /app/Team2/engine_server.py &
  else
    echo "WARNING: Python executable not found; engine server will not start"
  fi
fi

exec "$@"