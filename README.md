# fix-my-elo

FixMyElo, ML Projects 2025-2026.

This README covers the full-stack app only: architecture, local setup, and deployment.
Model research code lives in [Team1/](Team1/) and [Team2/](Team2/).

## Architecture

Three services, split backend-for-frontend:

| Service | Path | Port | Stack |
| --- | --- | --- | --- |
| `frontend` | [fme-app/](fme-app/) | 3000 | Next.js 16, React 19, Tailwind 4 |
| `api` | [backend/api/](backend/api/) | 8000 | FastAPI, no torch |
| `engine` | [backend/services/engine_service/](backend/services/engine_service/) | 8001 | FastAPI, PyTorch, MCTS |

```
browser ──POST /agent_move/──▶ api ──POST {ENGINE_URL}/move──▶ engine
```

[MainViewer.jsx](fme-app/components/main/MainViewer.jsx) renders the board, parses
PGNs, and posts the current FEN to the api.
[move.py](backend/api/call_engine/move.py) validates the FEN, applies defaults for
`numSimulations` and `temperature`, and proxies to the engine.
[engine_server.py](backend/services/engine_service/engine_server.py) loads
`SLPolicyValueNetwork` from a `.pth` checkpoint, runs MCTS, and returns the move in UCI
and SAN with per-move visit counts, evals, and priors.

The engine is the only GPU-bound service. It runs on Modal in production and as a CPU
container locally.

## Running locally

Requires Docker with Compose v2.

### Model weights

`.pth` files are gitignored, so download the checkpoint into the engine service folder
before starting anything:

```bash
curl -L \
  https://huggingface.co/Khushi-Malik/fix-my-elo/resolve/main/SL_trained_stockfish_trained.pth \
  -o backend/services/engine_service/SL_trained_stockfish_trained.pth
```

The file is 370 MB and the repo is public, so no Hugging Face token is needed. Keep the
filename as is or set `MODEL_PATH` to match. The engine exits on startup without it.

### Fully local

Runs `frontend`, `api`, and `engine_server` on your machine:

```bash
docker compose -f compose.local.yaml up --build
```

Open http://localhost:3000. The api is at http://localhost:8000 and the engine at
http://localhost:8001/health. The first build installs the CPU-only torch wheel, so it
takes a while.

### Remote engine

Runs `frontend` and `api` locally against the engine deployed on Modal. Needs a root
[.env](.env) with Modal credentials and `ENGINE_URL`:

```bash
docker compose up --build
```

### Individual services

From `backend/` with a virtualenv:

```bash
pip install -r api_requirements.txt
uvicorn api.main:app --reload --port 8000

pip install -r engine_requirements.txt   # plus torch, CPU wheel
uvicorn services.engine_service.engine_server:app --reload --port 8001
```

From `fme-app/`:

```bash
npm install
npm run dev
```

[engine_move_cli.py](backend/services/engine_service/engine_move_cli.py) runs a single
move from the command line.

## Configuration

Env files live at the repo root, `.env` and `.env.local`, and in
[fme-app/.env.local](fme-app/.env.local). Compose auto-loads root `.env` for `${VAR}`
substitution.

### api

| Variable | Purpose |
| --- | --- |
| `ENGINE_URL` | Base URL of the engine. `/move` is appended in code. |
| `FRONTEND_URL` | Added to the CORS allowlist alongside `http://localhost:3000`. |
| `AGENT_MOVE_DEFAULT_NUM_SIM` | MCTS simulations when the frontend omits it. Default 120. |
| `AGENT_MOVE_DEFAULT_TEMP` | Sampling temperature. Default 0.0. |
| `AGENT_MOVE_TIMEOUT_MS` | Upstream engine timeout. Default 150000. |
| `AGENT_MOVE_MIN_TIMEOUT_MS` | Floor applied to the timeout override. Default 100000. |

### engine

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Checkpoint path. Default `./services/engine_service/SL_trained_stockfish_trained.pth`. |
| `C_PUCT` | MCTS exploration constant. Default 1.0. |
| `DIRICHLET_ALPHA` / `DIRICHLET_EPSILON` | Root noise. Defaults 0.3 and 0.25. |
| `NUM_SIM_DEFAULT` / `TEMP_DEFAULT` | Engine-side fallbacks. |

### frontend

| Variable | Purpose |
| --- | --- |
| `BACKEND_API_URL` | Base URL of the api, no trailing slash. |
| `NEXT_PUBLIC_ENGINE_REQUEST_TIMEOUT_MS` | Client-side abort timeout. Default 170000. |
| `NEXT_PUBLIC_DEFAULT_NUM_SIMULATIONS` | Simulations requested per move. Default 120. |

Next.js inlines `NEXT_PUBLIC_*` variables at build time, so pass them as build args for
Docker builds. The board runs client-side, so its api URL must be reachable from the
browser, not just from inside the compose network.

### modal

| Variable | Purpose |
| --- | --- |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Credentials for `modal deploy`. |

## Deployment

### Engine on Modal

[modal_api.py](backend/services/engine_service/modal_api.py) builds the image from
[modal.Dockerfile](backend/modal.Dockerfile) and serves the same FastAPI app used
locally on a T4, with a 600s timeout and a 60s scaledown window.

Deployment is containerised.
[runtime.modal.Dockerfile](backend/runtime.modal.Dockerfile) installs the Modal CLI and
runs `modal deploy --env=FME-engine services/engine_service/modal_api.py`, which is what
the `modal` service in [compose.yaml](compose.yaml) does. Bring it up with valid Modal
tokens to deploy, then set the URL Modal prints as `ENGINE_URL` on the api.

The checkpoint reaches the image through `COPY . .`, so download the weights before
deploying and redeploy when they change.
[backend/.dockerignore](backend/.dockerignore) currently excludes `**/*.pth`, which
blocks that copy and leaves the deployed engine without a checkpoint.

### api and frontend

The api ships from [backend/Dockerfile](backend/Dockerfile) on python:3.13-slim as a
non-root user running `uvicorn api.main:app` on 8000. The frontend ships from
[fme-app/Dockerfile](fme-app/Dockerfile), a multi-stage node:24-alpine build running
`npm start` on 3000.

Build for the target architecture when it differs from your machine:

```bash
docker build --platform=linux/amd64 -t fme-api ./backend
docker build --platform=linux/amd64 -t fme-frontend ./fme-app
```

Set `FRONTEND_URL` on the api to the deployed frontend origin or CORS will block the
browser.
