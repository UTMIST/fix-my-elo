from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional
import os
import math
import chess
import httpx

router = APIRouter(prefix="/agent_move")


class AgentMoveRequest(BaseModel):
    fen: Optional[str] = None
    numSimulations: Optional[float] = None
    temperature: Optional[float] = None


class EnginePayload(BaseModel):
    fen: str
    numSimulations: int
    temperature: float


@router.get("/")
def test():
    return Response("routed to agent-move")


@router.post("/")
async def move_request(req: AgentMoveRequest):
    fen = req.fen.strip() if req.fen else None

    if not fen:
        return JSONResponse({"error": "fen is required"}, status_code=400)

    try:
        board = chess.Board(fen)
    except Exception:
        return JSONResponse({"error": "Invalid FEN"}, status_code=400)
    if not board.is_valid():
        return JSONResponse({"error": "Invalid FEN"}, status_code=400)

    num_simulations = (
        max(1, math.floor(req.numSimulations))
        if req.numSimulations is not None and math.isfinite(req.numSimulations)
        else 120
    )

    temperature = (
        float(req.temperature)
        if req.temperature is not None and math.isfinite(req.temperature)
        else 0.0
    )

    timeout_env = _to_float(os.environ.get("AGENT_MOVE_TIMEOUT_MS"))
    timeout_ms = max(100000, timeout_env) if timeout_env is not None else 150000

    engine_url = os.environ.get("TEAM2_ENGINE_URL") or os.environ.get("TEAM2_INFERENCE_URL")
    if not engine_url:
        return JSONResponse(
            {
                "error": "TEAM2_ENGINE_URL is not configured",
                "details": "Set TEAM2_ENGINE_URL to your Modal /move endpoint.",
            },
            status_code=500,
        )

    engine_response = await call_engine_server(
        engine_url,
        EnginePayload(
            fen=fen,
            numSimulations=num_simulations,
            temperature=temperature,
        ),
        timeout_ms,
    )

    return JSONResponse(engine_response["body"], status_code=engine_response["status"])


async def call_engine_server(url: str, payload: EnginePayload, timeout_ms: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                content=payload.model_dump_json(),
            )

        raw = response.text
        body: dict = {}

        if raw:
            try:
                body = response.json()
            except Exception:
                body = {
                    "error": "Engine server returned non-JSON response",
                    "details": raw,
                }

        return {"ok": response.is_success, "status": response.status_code, "body": body}
    except Exception as error:
        return {
            "ok": False,
            "status": 504,
            "body": {
                "error": "Engine server request failed",
                "details": str(error),
            },
        }


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
