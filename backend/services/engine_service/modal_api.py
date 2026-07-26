import modal
from dotenv import load_dotenv

# image = (
#     modal.Image.debian_slim(python_version="3.13")
#     # .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cpu")
#     .pip_install("torch==2.11.0")
#     .pip_install_from_requirements(requirements_txt="engine_requirements.txt")
#     .add_local_dir("./services", "/root/services", ignore=["**/__pycache__"], copy=True)
#     .add_local_dir("./api", "/root/api", ignore=["**/__pycache__"], copy=True)
#     .env(
#         {"MODEL_PATH": "/root/services/engine_service/SL_trained_stockfish_trained.pth"}
#     )
# )
image = (
    modal.Image.from_dockerfile("modal.Dockerfile")
)

app = modal.App("FME-engine", image=image)


@app.function(image=image, scaledown_window=60, timeout=600, gpu="T4")
@modal.asgi_app()
def modal_webapp():
    from services.engine_service.engine_server import app as engine_server

    return engine_server
