import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from api.call_engine.move import router as agent_move_router

load_dotenv(".env")

app = FastAPI()

app.include_router(agent_move_router)
origins = [os.getenv("FRONTEND_URL"), "http://localhost:3100", "https://qmjv51qr-3100.use.devtunnels.ms"]
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
