from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel
from typing import Optional
import chess

router = APIRouter(
    prefix="/agent_move"
)

class AgentMoveRequest(BaseModel):
    fen: Optional[str]
    numSimulations: Optional[int]
    temperature: Optional[int]


@router.get("/")
def test():
    return Response("routed to agent-move")

@router.post("/")
async def move_request(req: AgentMoveRequest):
    
    try:
        fen  = chess.Board(req.fen)
    except:
        raise HTTPException(400, "invalid FEN")
    if not fen.is_valid():
        raise HTTPException(400, "invalid FEN")
    