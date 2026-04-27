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
    FILE_SIZE=$(stat -f%z "$TEAM2_MODEL_PATH" 2>/dev/null || stat -c%s "$TEAM2_MODEL_PATH" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 1000 ]; then
      echo "ERROR: Downloaded file too small (${FILE_SIZE} bytes). May be an error page."
      rm -f "$TEAM2_MODEL_PATH"
      exit 1
    fi
    
    # Check for common error signatures in downloaded file
    if head -c 100 "$TEAM2_MODEL_PATH" | grep -q "<!DOCTYPE\|<html\|404\|error"; then
      echo "ERROR: Downloaded file appears to be HTML (likely an error page, not a PyTorch checkpoint)"
      rm -f "$TEAM2_MODEL_PATH"
      exit 1
    fi
    
    echo "Model downloaded successfully (${FILE_SIZE} bytes)"
  else
    echo "Team2 model already present at ${TEAM2_MODEL_PATH}"
  fi
fi

exec "$@"