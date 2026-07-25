from fastapi import FastAPI
from fastapi.responses import Response
from api.agent_move.move import router as agent_move_router

app = FastAPI()

app.include_router(agent_move_router)


@app.get("/")
def index():
    return Response("Running!")
