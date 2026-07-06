from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Annotated
import os
import chess

router = APIRouter(prefix="/agent_move")


class AgentMoveRequest(BaseModel):
    fen: Optional[str]
    numSimulations: Optional[Annotated[int, Field(gt=2)]]
    temperature: Optional[Annotated[float, Field(ge=0)]]


@router.get("/")
def test():
    return Response("routed to agent-move")


@router.post("/")
async def move_request(req: AgentMoveRequest):

    try:
        fen = chess.Board(req.fen)
    except:
        raise HTTPException(400, "invalid FEN")
    if not fen.is_valid():
        raise HTTPException(400, "invalid FEN")

    timeoutMs = os.environ.get("AGENT_MOVE_TIMEOUT_MS", default=1000)
    engineUrl = os.envrion.get("TEAM2_ENGINE_URL", None)

    if not engineUrl:
        raise HTTPException(500, "Engine url not set")


async def callEngineServer(req: AgentMoveRequest):
    pass
