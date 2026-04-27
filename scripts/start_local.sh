#!/usr/bin/env bash
set -euo pipefail

# start_local.sh
# Creates a local virtualenv, installs Python deps, optionally downloads a
# Hugging Face checkpoint into Team2/model_files, updates fme-app/.env.local,
# and starts the Next dev server.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
TEAM2_DIR="$REPO_ROOT/Team2"
MODEL_DIR="$TEAM2_DIR/model_files"
FME_APP_DIR="$REPO_ROOT/fme-app"
ENV_FILE="$FME_APP_DIR/.env.local"

HF_URL="${1:-${TEAM2_MODEL_URL:-}}"
MODEL_NAME="${2:-hf_model.pth}"

echo "Repo root: $REPO_ROOT"

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  if [ -f "$TEAM2_DIR/requirements.txt" ]; then
    echo "Installing Team2 Python requirements"
    pip install -r "$TEAM2_DIR/requirements.txt"
  else
    echo "Warning: $TEAM2_DIR/requirements.txt not found; skipping pip install"
  fi
}

download_model() {
  if [ -z "$HF_URL" ]; then
    echo "No HF URL provided; skipping model download. Provide URL as first arg or set TEAM2_MODEL_URL."
    return 0
  fi
  mkdir -p "$MODEL_DIR"
  dest="$MODEL_DIR/$MODEL_NAME"
  echo "Downloading model from $HF_URL to $dest"
  curl -L --fail -o "$dest" "$HF_URL"
  filesize=$(wc -c <"$dest" | tr -d ' ')
  if [ "$filesize" -lt 50000 ]; then
    echo "Downloaded file looks small ($filesize bytes). Check the URL or remove the file: $dest"
    return 1
  fi
  echo "Model downloaded: $dest ($filesize bytes)"
}

set_env_var() {
  key="$1"; val="$2"; file="$3"
  mkdir -p "$(dirname "$file")"
  if [ -f "$file" ]; then
    if grep -q "^${key}=" "$file"; then
      tmpfile="${file}.tmp"
      awk -v k="$key" -v v="$val" 'BEGIN{found=0} $0 ~ "^"k"="{print k"="v; found=1; next} {print} END{if(!found) print k"="v}' "$file" > "$tmpfile"
      mv "$tmpfile" "$file"
      return
    fi
  fi
  echo "${key}=${val}" >> "$file"
}

main() {
  ensure_venv

  if [ -n "$HF_URL" ]; then
    download_model || true
  fi

  MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
  # If explicit model file not present, try to leave env alone
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file $MODEL_PATH not found. You can place your checkpoint there or re-run with a HF URL."
  fi

  # Update fme-app/.env.local
  echo "Updating $ENV_FILE"
  set_env_var "TEAM2_MODEL_PATH" "$MODEL_PATH" "$ENV_FILE"
  set_env_var "TEAM2_PYTHON" "$VENV_DIR/bin/python" "$ENV_FILE"

  echo "To start the frontend dev server, the script will run 'npm run dev' in $FME_APP_DIR"
  echo "If this is the first time, you may want to run 'npm ci' in $FME_APP_DIR first."

  cd "$FME_APP_DIR"
  if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies (npm ci)"
    npm ci
  fi

  echo "Starting Next dev server"
  npm run dev
}

main "$@"
