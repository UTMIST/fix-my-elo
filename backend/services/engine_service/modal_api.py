import os

import modal
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "services/engine_service/model_weights/SL_trained_stockfish_trained.pth",
)
if not MODEL_PATH.startswith("/"):
    MODEL_PATH = f"/app/{MODEL_PATH}"

# Engine config is a file path and a handful of search params, not credentials,
# so it goes in the image environment rather than a modal.Secret. Unset keys are
# dropped so engine_server.py falls back to its own defaults instead of trying to
# parse an empty string.
engine_env = {
    key: value
    for key, value in {
        "MODEL_PATH": MODEL_PATH,
        "C_PUCT": os.getenv("C_PUCT"),
        "DIRICHLET_ALPHA": os.getenv("DIRICHLET_ALPHA"),
        "DIRICHLET_EPSILON": os.getenv("DIRICHLET_EPSILON"),
        "MCTS_BATCH_SIZE": os.getenv("MCTS_BATCH_SIZE"),
        "NUM_SIM_DEFAULT": os.getenv("NUM_SIM_DEFAULT"),
        "TEMP_DEFAULT": os.getenv("TEMP_DEFAULT"),
    }.items()
    if value is not None
}

# .env() is appended after the Dockerfile layers, so changing a value here
# rebuilds only that last layer and not the multi-GB COPY of the weights.
image = modal.Image.from_dockerfile("modal.Dockerfile").env(engine_env)

app = modal.App("FME-engine", image=image)


@app.function(
    image=image,
    scaledown_window=60,
    timeout=600,
    gpu="T4",
)
@modal.asgi_app()
def modal_webapp():
    from services.engine_service.engine_server import app as engine_server

    return engine_server
