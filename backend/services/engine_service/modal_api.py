from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import logging
import modal
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "Team2" if (BASE_DIR / "Team2").exists() else BASE_DIR
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from requirements.agent import Agent
from requirements.SLPolicyValueGPU import SLPolicyValueNetwork


MODEL_URL = os.environ.get("MODEL_URL")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/tmp/fix-my-elo-model.pth"))
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "1800"))
DEFAULT_NUM_SIM = int(os.environ.get("NUM_SIM_DEFAULT", "120"))
DEFAULT_TEMP = float(os.environ.get("TEMP_DEFAULT", "0.0"))
C_PUCT = float(os.environ.get("C_PUCT", "1.0"))
DIRICHLET_ALPHA = float(os.environ.get("DIRICHLET_ALPHA", "0.3"))
DIRICHLET_EPSILON = float(os.environ.get("DIRICHLET_EPSILON", "0.25"))

app = modal.App("fix-my-elo-engine")


class MoveRequest(BaseModel):
    fen: str
    numSimulations: int | None = None
    temperature: float | None = None


_AGENT: Agent | None = None


def _download_model(model_url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    import requests

    logging.info("Downloading model from %s to %s", model_url, destination)
    with requests.get(model_url, stream=True, timeout=MODEL_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            total = 0
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file_handle.write(chunk)
                    total += len(chunk)
                    logging.info("Wrote chunk, total bytes so far: %d", total)

    size = destination.stat().st_size
    logging.info("Finished download, file size: %d bytes", size)
    if size < 1000:
        raise RuntimeError(f"Downloaded model file is too small: {size} bytes")

    return destination


def _load_agent() -> Agent:
    global _AGENT

    if _AGENT is not None:
        return _AGENT

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is not configured")

    model_path = _download_model(MODEL_URL, MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    model = SLPolicyValueNetwork().to(device)

    logging.info("Loading checkpoint from %s", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _AGENT = Agent(
        policy_value_network=model,
        c_puct=C_PUCT,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
    )
    logging.info("Agent initialized successfully")
    return _AGENT


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "fastapi>=0.115.0",
        "requests>=2.32.0",
        "python-chess>=1.999",
        "numpy>=1.26.0",
        "torch",
    ).add_local_dir(str(BASE_DIR), remote_path="/root/Team2"),
    secrets=[modal.Secret.from_name("fix-my-elo-team2")],
    memory=4096,
    cpu=2,
    timeout=1800,
)
@modal.asgi_app()
def api() -> FastAPI:
    web_app = FastAPI()

    @web_app.on_event("startup")
    async def _startup_preload() -> None:
        logging.info("Startup: scheduling background agent preload")
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _load_agent)
        except Exception:
            logging.exception("Failed to schedule agent preload")

    @web_app.post("/force-load")
    async def force_load() -> dict[str, object]:
        try:
            agent = _load_agent()
            return {
                "ok": True,
                "agent_loaded": _AGENT is not None,
                "model_path": str(MODEL_PATH),
                "model_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
            }
        except Exception as exc:
            logging.exception("Force load failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @web_app.get("/model-info")
    async def model_info() -> dict[str, object]:
        info: dict[str, object] = {
            "model_url": MODEL_URL,
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "model_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
            "agent_loaded": _AGENT is not None,
        }
        return info

    @web_app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @web_app.post("/move")
    async def move(request: MoveRequest) -> dict[str, object]:
        fen = request.fen.strip()
        if not fen:
            raise HTTPException(status_code=400, detail="fen is required")

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

        if board.is_game_over(claim_draw=True):
            raise HTTPException(
                status_code=400,
                detail={"error": "Game already over", "result": board.result(claim_draw=True)},
            )

        num_simulations = (
            max(1, int(request.numSimulations))
            if request.numSimulations is not None
            else DEFAULT_NUM_SIM
        )
        temperature = float(request.temperature) if request.temperature is not None else DEFAULT_TEMP

        agent = _load_agent()
        move_uci = str(
            agent.select_move(
                game_state=board,
                num_simulations=num_simulations,
                temperature=temperature,
                debug=False,
            )
        )
        move_obj = chess.Move.from_uci(move_uci)

        return {
            "move": move_uci,
            "san": board.san(move_obj),
            "fen": fen,
            "numSimulations": num_simulations,
        }

    return web_app