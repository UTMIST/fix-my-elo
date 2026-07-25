import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from api.call_engine.move import router as agent_move_router

# fastapi dev / uvicorn do NOT read .env files on their own, so os.environ never
# sees them. Load backend/.env explicitly (path is relative to this file, so it
# works regardless of the working directory). No-op in Docker where env is preset.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI()

app.include_router(agent_move_router)
origins = [os.getenv("FRONTEND_URL"), "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return Response("Running!")
