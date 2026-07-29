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

`.pth` files are gitignored, so download the checkpoints before starting anything:

**[Google Drive: model weights](https://drive.google.com/drive/folders/16eUW6h6a5prGp_lHqUCJzWj5A6eL8Pk7)**

Put them in
[backend/services/engine_service/model_weights/](backend/services/engine_service/model_weights/):

```
backend/services/engine_service/model_weights/
├── SL_trained_stockfish_trained.pth
└── lab_trained_epoch_1.pth
```

Each file is around 370 MB. Keep the filenames as they are and point `MODEL_PATH` at the
one you want, or rename and set `MODEL_PATH` to match. The engine exits on startup if the
path does not resolve, logging the path it tried.

`MODEL_PATH` is resolved relative to the working directory, which is `/app` in every
container, so the relative form in [.env](.env) works both locally and on Modal.

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

Every tool in the stack looks for a file named `.env`, so that is the only root env file
that does anything. [.env.local](.env.local) is the tracked reference copy, holding
local-dev values and a comment for every variable. `.env` itself is gitignored, which is
where the Modal tokens go.

To start from the reference values, copy it over:

```bash
cp .env.local .env
```

Nothing reads `.env.local` under that name, so editing it alone has no effect. There is
also a separate [fme-app/.env.local](fme-app/.env.local) for the frontend, which `next
dev` does load automatically.

Compose auto-loads root `.env`, but only to substitute `${VAR}` inside the compose file.
It does not inject anything into containers. Every variable a service needs must also
appear in that service's `environment:` block, which is why the blocks in
[compose.yaml](compose.yaml) and [compose.local.yaml](compose.local.yaml) restate them.
Adding a new engine variable means adding it in three places: `.env`, the service's
`environment:` block, and the dict in
[modal_api.py](backend/services/engine_service/modal_api.py) if the deployed engine needs
it too.

Of the two services, only [api/main.py](backend/api/main.py) calls `load_dotenv`, and
`.env` is excluded by [backend/.dockerignore](backend/.dockerignore) anyway, so even there
it is a no-op inside containers. The engine reads real environment variables only, so
running it with bare `uvicorn` outside Docker picks up none of `.env` unless you export
the values into your shell first.
[modal_api.py](backend/services/engine_service/modal_api.py) also calls `load_dotenv`, but
it runs at deploy time on your machine rather than inside a container.

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
| `MODEL_PATH` | Checkpoint path. Default `./services/engine_service/model_weights/SL_trained_stockfish_trained.pth`. |
| `C_PUCT` | MCTS exploration constant. Default 1.0. |
| `DIRICHLET_ALPHA` / `DIRICHLET_EPSILON` | Root noise. Defaults 0.3 and 0.25. |
| `MCTS_BATCH_SIZE` | Leaf positions queued before one batched forward pass. Default 8. |
| `NUM_SIM_DEFAULT` / `TEMP_DEFAULT` | Engine-side fallbacks. |

All of these are read once at import time in
[engine_server.py](backend/services/engine_service/engine_server.py), when the module
constructs the `Agent`, so changing one needs a restart.

`MCTS_BATCH_SIZE` is an upper bound that is rarely reached. MCTS force-flushes its queue
whenever selection walks back to a still-pending leaf, and since every queued leaf
consumes one simulation, the queue can never exceed the simulation count. Any value at or
above `NUM_SIM_DEFAULT` is therefore unreachable and behaves identically. Raising it
trades fewer, larger forward passes against staler tree statistics during selection.

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

The `modal` service also receives the engine variables above. It does not use them
itself: [modal_api.py](backend/services/engine_service/modal_api.py) reads them at deploy
time and bakes them into the image environment, which is how they reach the deployed
engine.

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

Engine config is baked into the image with `Image.env()` rather than passed as a
`modal.Secret`, since a checkpoint path and a few search constants are not credentials.
`.env()` is applied after the Dockerfile layers, so changing a value rebuilds only that
final layer and does not re-copy the weights. The tradeoff is that engine config changes
need a redeploy to take effect.

To read the effective config of a running engine, check its startup logs. The engine logs
the resolved checkpoint path and MCTS batch size as it constructs the `Agent`.

#### Modal environments

The deploy targets the `FME-engine` environment, not the default `main`:

```bash
modal environment list      # main is active; FME-engine is where the app lives
modal app list --env FME-engine
```

The dashboard opens on the active environment, so the app, its functions, and its
containers are all invisible until you switch the environment selector to `FME-engine`.
That environment's web suffix is also what produces the `fme-engine` segment in
`ENGINE_URL`, which is a quick way to tell which environment a URL belongs to.

The Containers panel will usually look empty even in the right environment. With a 60s
scaledown window the app runs at zero containers between requests, and that panel shows
only what is running right now, not history. Use the app's function view and logs for past
activity.

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
