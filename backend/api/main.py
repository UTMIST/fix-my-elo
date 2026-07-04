from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional
from agent_move.move import router as agent_move_router

app = FastAPI()

app.include_router(agent_move_router)

@app.get("/")
def index():
    return Response("Running!")