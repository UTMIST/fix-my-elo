import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import chess
from services.engine_service.requirements.agent import Agent
from services.engine_service.requirements.SLPolicyValueGPU import SLPolicyValueNetwork
from api.call_engine.move import EnginePayload
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_agent(
    model_path: str, c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float
) -> Agent:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    model = SLPolicyValueNetwork().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

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

    return Agent(
        policy_value_network=model,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )


def parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


MODEL_PATH = os.getenv(
    "MODEL_PATH", "./services/engine_service/SL_trained_stockfish_trained.pth"
)
C_PUCT = parse_float("C_PUCT", 1.0)
DIRICHLET_ALPHA = parse_float("DIRICHLET_ALPHA", 0.3)
DIRICHLET_EPSILON = parse_float("DIRICHLET_EPSILON", 0.25)
DEFAULT_NUM_SIM = parse_int("NUM_SIM_DEFAULT", 120)
DEFAULT_TEMP = parse_float("TEMP_DEFAULT", 0.0)

app = FastAPI()


try:
    AGENT = load_agent(MODEL_PATH, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON)
except Exception as exc:
    print(f"[engine_server] Failed to load agent: {exc}", file=sys.stderr)
    sys.exit(1)


@app.get("/health")
def get():
    return JSONResponse("running!", 200)


@app.post("/move")
def post(payload: EnginePayload):

    fen = payload.fen
    if not fen:
        raise HTTPException(400, "error: fen is required")

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise HTTPException(400, exc)

    if board.is_game_over(claim_draw=True):
        raise HTTPException(
            400, {"error": "Game already over", "result": board.result(claim_draw=True)}
        )

    try:
        num_sim = int(payload.numSimulations)
    except Exception as e:
        raise HTTPException(400, e)
    if num_sim < 1:
        num_sim = 1

    try:
        temp = float(payload.temperature)
    except Exception as e:
        raise HTTPException(400, e)

    try:
        engine_reponse = AGENT.select_move(
            game_state=board,
            num_simulations=num_sim,
            temperature=temp,
            debug=True,
        )
        move_uci, combined = engine_reponse[0], engine_reponse[1]

        move_obj = chess.Move.from_uci(move_uci)
        san = board.san(move_obj)
    except Exception as exc:
        raise HTTPException(500, {"error": f"Engine failure: {exc}"})

    # `combined` is a list of (move, count, eval, prior) tuples where the numeric
    # fields are numpy scalars (float32/float64) that json can't serialize. Convert
    # to native Python types so JSONResponse can encode them.
    counts = [
        {
            "move": str(move),
            "count": float(count),
            "eval": float(evaluation),
            "prior": float(prior),
        }
        for move, count, evaluation, prior in combined
    ]

    return JSONResponse(
        {
            "move": move_uci,
            "counts": counts,
            "san": san,
            "fen": fen,
            "numSimulations": num_sim,
        },
        200,
    )
