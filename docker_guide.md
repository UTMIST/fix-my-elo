# Thing to know

1. Docker has 2 types of variables: interpolation (buildtime) and runtime injection (runtime)

- Interpolation is what gets fed into the build recipe of a container.
- Runtime injection is what gets fed into the container once the container is up and running.

2. You need to specify the `.env` file you want compose to for buildtime, otherwise it defaults to `.env`.

- When compose parses lines like `${VARIABLE}`, by default it looks up `.env` and finds the value for `VARIABLE`.
- When running `docker compose`, by default it runs a file called `compose.yaml`.
- To specify the compose file and env file, use `docker compose -f (compose file) --env-file (env file) up --build`.

3. The `env_file` section of a service definition merely specifies the env file to be read for runtime injection.

- You first need to parse all `${VARIABLE}` statements in buildtime first, so you also need to specify the env file with the `--env-file` flag in the CLI.

# Passing env variables into containers

Requirements:

1. An env file containing the variable names and values
2. A dockerfile for the container
3. A compose file that starts the container (service), feeding env variables

### Example: passing `MODEL_PATH` in `local.cloudflare.env` to `engine_server`

Assume in `local.cloudflare.env`, `MODEL_PATH=<path>`, and that `engine_server.py` in `engine_server` can parse env variables.

- Since `MODEL_PATH` is only used to load the model ONCE the container has already launched, we should consider runtime injection (environment).
- In compose, add `environment: MODEL_PATH=${MODEL_PATH}`
- Launch the container with `docker compose -f compose.local.cloudflare.yaml --env-file local.cloudflare.env up -d engine_server --build`.

### Caveat for Next.js

- You need to pass all public variables the browser should be able to see as `NEXT_PUBLIC_<name>`, inside `args` of compose, as client-side varibles (variables required by the browser) are fixed at buildtime.
- You need to specify `PORT` in `environment` in compose, because Next.js reads the only after startup.

### Example: passing `BACKEND_API_URL` in `local.cloudflare.env` to `frontend`

Assume in `local.cloudflare.env`, `BACKEND_API_URL=<url>`, and that `MainViewer.jsx` in `fme-app` can parse env variables.

- Since `BACKEND_API_URL` is required by the client-side browser to make requests to the API, it needs to be baked into the client-side JS during buildtime, hence we consider interpolation (args).
- In compose, add `args: NEXT_PUBLIC_BACKEND_API_URL=${BACKEND_API_URL}`.
- In dockerfile for frontend, add `ARG NEXT_PUBLIC_BACKEND_API_URL` before `npm run build`.
- Launch the container with `docker compose -f compose.local.cloudflare.yaml --env-file local.cloudflare.env up -d frontend --build`.
