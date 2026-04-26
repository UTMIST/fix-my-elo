import argparse
import json
import os
import sys

import chess
import torch

from agent import Agent
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork


def load_agent(model_path: str, c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float) -> Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLPolicyValueNetwork().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return Agent(
        policy_value_network=model,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Return one engine move for a given FEN as JSON.")
    parser.add_argument("--fen", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-simulations", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    args = parser.parse_args()

    try:
        board = chess.Board(args.fen)
    except ValueError as e:
        print(json.dumps({"error": f"Invalid FEN: {e}"}))
        return 1

    if board.is_game_over(claim_draw=True):
        print(json.dumps({"error": "Game already over", "result": board.result(claim_draw=True)}))
        return 1

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model file not found: {model_path}"}))
        return 1

    try:
        agent = load_agent(
            model_path=model_path,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )

        move_uci = str(
            agent.select_move(
                game_state=board,
                num_simulations=args.num_simulations,
                temperature=args.temperature,
                debug=False,
            )
        )

        move_obj = chess.Move.from_uci(move_uci)
        san = board.san(move_obj)

        print(json.dumps({
            "move": move_uci,
            "san": san,
            "fen": args.fen,
            "numSimulations": args.num_simulations,
        }))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Engine failure: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
