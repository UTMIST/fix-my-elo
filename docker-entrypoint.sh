#!/bin/sh
set -eu

if [ -n "${TEAM2_MODEL_URL:-}" ]; then
  mkdir -p "$(dirname "$TEAM2_MODEL_PATH")"

  if [ ! -s "$TEAM2_MODEL_PATH" ]; then
    echo "Downloading Team2 model from ${TEAM2_MODEL_URL} to ${TEAM2_MODEL_PATH}"
    curl -L --fail --retry 3 --retry-delay 2 "$TEAM2_MODEL_URL" -o "$TEAM2_MODEL_PATH"
  else
    echo "Team2 model already present at ${TEAM2_MODEL_PATH}"
  fi
fi

exec "$@"