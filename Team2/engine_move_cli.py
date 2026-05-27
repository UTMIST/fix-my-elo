import argparse
import json
import os
import sys

import chess
import torch

from agent import Agent
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork


def load_agent(model_path: str, c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float) -> Agent:
    """Load a checkpoint and create an Agent instance."""
    import traceback
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLPolicyValueNetwork().to(device)

    try:
        # Validate file exists and is not empty
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        file_size = os.path.getsize(model_path)
        if file_size < 1000:
            raise ValueError(f"Model file too small ({file_size} bytes). May not be a valid checkpoint.")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the entire dict is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load model weights
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}") from e
    
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
        import traceback as tb
        
        agent = load_agent(
            model_path=model_path,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )

        output = agent.select_move(
                game_state=board,
                num_simulations=args.num_simulations,
                temperature=args.temperature,
                debug=True,
            )

        move_uci = str(output[0])
        explored_moves = output[1] # each item in this array looks like [move uci, #times explored, expected value, policy prior]

        move_obj = chess.Move.from_uci(move_uci)
        san = board.san(move_obj)

        print(json.dumps({
            "move": move_uci,
            "san": san,
            "fen": args.fen,
            "numSimulations": args.num_simulations,
            "explored_moves": explored_moves
        }))
        return 0
    except Exception as e:
        import traceback as tb
        error_trace = tb.format_exc()
        print(json.dumps({"error": f"Engine failure: {e}", "trace": error_trace}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
