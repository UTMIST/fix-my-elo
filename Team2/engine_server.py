import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

from agent import Agent
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork


def load_agent(model_path: str, c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float) -> Agent:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


MODEL_PATH = os.getenv("TEAM2_MODEL_PATH", "/app/Team2/model_files/model.pth")
C_PUCT = parse_float("TEAM2_C_PUCT", 1.0)
DIRICHLET_ALPHA = parse_float("TEAM2_DIRICHLET_ALPHA", 0.3)
DIRICHLET_EPSILON = parse_float("TEAM2_DIRICHLET_EPSILON", 0.25)
DEFAULT_NUM_SIM = parse_int("TEAM2_NUM_SIM_DEFAULT", 120)
DEFAULT_TEMP = parse_float("TEAM2_TEMP_DEFAULT", 0.0)


try:
    AGENT = load_agent(MODEL_PATH, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON)
except Exception as exc:
    print(f"[engine_server] Failed to load agent: {exc}", file=sys.stderr)
    sys.exit(1)


class EngineHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/move":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw) if raw else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        fen = str(payload.get("fen", "")).strip()
        if not fen:
            self._send_json(400, {"error": "fen is required"})
            return

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            self._send_json(400, {"error": f"Invalid FEN: {exc}"})
            return

        if board.is_game_over(claim_draw=True):
            self._send_json(400, {"error": "Game already over", "result": board.result(claim_draw=True)})
            return

        try:
            num_sim = int(payload.get("numSimulations", DEFAULT_NUM_SIM))
        except Exception:
            num_sim = DEFAULT_NUM_SIM
        if num_sim < 1:
            num_sim = 1

        try:
            temp = float(payload.get("temperature", DEFAULT_TEMP))
        except Exception:
            temp = DEFAULT_TEMP

        try:
            move_uci = str(
                AGENT.select_move(
                    game_state=board,
                    num_simulations=num_sim,
                    temperature=temp,
                    debug=False,
                )
            )
            move_obj = chess.Move.from_uci(move_uci)
            san = board.san(move_obj)
        except Exception as exc:
            self._send_json(500, {"error": f"Engine failure: {exc}"})
            return

        self._send_json(
            200,
            {
                "move": move_uci,
                "san": san,
                "fen": fen,
                "numSimulations": num_sim,
            },
        )

    def log_message(self, format: str, *args) -> None:
        # silence default HTTP server logs
        return


def main() -> int:
    host = os.getenv("TEAM2_ENGINE_HOST", "127.0.0.1")
    port = parse_int("TEAM2_ENGINE_PORT", 8001)
    server = ThreadingHTTPServer((host, port), EngineHandler)
    print(f"[engine_server] Listening on http://{host}:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
