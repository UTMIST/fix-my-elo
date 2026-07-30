"""Play against the trained policy/value network from the terminal.

Runs as a normal python script so multiprocessing inside MCTS works properly
(unlike the notebook version).

Examples:
    python Team2/play_against_model.py human
    python Team2/play_against_model.py human --color black --sims 800
    python Team2/play_against_model.py stockfish --depth 15
    python Team2/play_against_model.py batch --games 2 --out pgn_files/demo.pgn
"""

import argparse
import os
import sys

# agent.py / monte_carlo_tree_search.py / minimax.py import `Team2.data_processing`,
# so the repo root (the parent of Team2/) has to be on sys.path. This script lives in
# Team2/, so sys.path also needs Team2/ itself for `agent` and `model_files`.
TEAM2_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEAM2_DIR)
for path in (REPO_ROOT, TEAM2_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import chess
import torch

from agent import Agent
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork

DEFAULT_WEIGHTS = os.path.join(TEAM2_DIR, "model_weights", "lab_trained_epoch_1.pth")


def build_agent(weights_path, device):
    model = SLPolicyValueNetwork().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return Agent(
        policy_value_network=model,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
    )


def model_move(agent, board, args, show_stats=True):
    """Ask the agent for a move, print its top lines, and push it."""
    move, stats = agent.select_move(
        game_state=board,
        num_simulations=args.sims,
        temperature=0,
        debug=True,
        mcts_policy_temperature=args.policy_temp,
        mcts_temperature=args.mcts_temp,
    )
    print(f"model plays {board.san(chess.Move.from_uci(move))} ({move})")
    if show_stats:
        for rank, (uci, visits, share, ev, prior) in enumerate(stats[:5], start=1):
            label = board.san(chess.Move.from_uci(uci)) + f" ({uci})"
            print(f"{rank:>2}  {label:<16}{int(visits):>8}{share:>8.1%}{ev:>+9.3f}{prior:>9.4f}")
    board.push_uci(move)
    return move


def play_human(agent, args):
    board = chess.Board()
    human_turn = args.color == "white"
    while not board.is_game_over():
        print(board, "\n")
        if human_turn:
            move = input("enter a move in UCI format (q to quit)\n").strip()
            if move == "q":
                break
            try:
                board.push_uci(move)
            except Exception:
                print("illegal or malformed move, try again")
                continue
            human_turn = not human_turn
        else:
            model_move(agent, board, args)
            human_turn = not human_turn
    print(board, "\n")
    print(board.result())


def play_stockfish(agent, args):
    board = chess.Board()
    for opening_move in args.opening:
        board.push_uci(opening_move)

    agent.stockfish.set_depth(args.depth)
    stockfish_turn = args.color == "black"  # agent takes the color it was given
    moves = []
    while not board.is_game_over():
        print(board, "\n")
        if stockfish_turn:
            agent.stockfish.set_fen_position(board.fen())
            print("stockfish eval:", agent.stockfish.get_evaluation()["value"] / 100)
            move = agent.stockfish.get_best_move()
            print(f"stockfish plays {board.san(chess.Move.from_uci(move))} ({move})")
            board.push_uci(move)
        else:
            move = model_move(agent, board, args)
        stockfish_turn = not stockfish_turn
        moves.append(move)
        print(" ".join(moves))
    print(board, "\n")
    print(board.result())


def play_batch(agent, args):
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(TEAM2_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    agent.stockfish.set_depth(args.depth)
    agent.agent_vs_stockfish(
        args.games,
        args.sims,
        out_path,
        mcts_policy_temperature=args.policy_temp,
        mcts_temperature=args.mcts_temp,
    )
    print("wrote", out_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["human", "stockfish", "batch"], nargs="?", default="human")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to the .pth checkpoint")
    parser.add_argument("--sims", type=int, default=500, help="MCTS simulations per move")
    parser.add_argument(
        "--policy-temp",
        dest="policy_temp",
        type=float,
        default=1.0,
        help="policy temperature, 1.0 means no temperature",
    )
    parser.add_argument(
        "--mcts-temp",
        dest="mcts_temp",
        type=float,
        default=1.0,
        help="UCT temperature, below 1.0 boosts top moves",
    )
    parser.add_argument("--color", choices=["white", "black"], default="white", help="color the opponent (you or stockfish) plays")
    parser.add_argument("--depth", type=int, default=15, help="stockfish search depth")
    parser.add_argument(
        "--opening",
        nargs="*",
        default=[],
        metavar="UCI",
        help="UCI moves to play out before the game starts, e.g. --opening e2e4 e7e5 f2f4",
    )
    parser.add_argument("--games", type=int, default=2, help="batch mode: number of games")
    parser.add_argument(
        "--out",
        default="pgn_files/agent_vs_stockfish.pgn",
        help="batch mode: pgn output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print("loading weights:", args.weights)
    agent = build_agent(args.weights, device)

    if args.mode == "human":
        play_human(agent, args)
    elif args.mode == "stockfish":
        play_stockfish(agent, args)
    else:
        play_batch(agent, args)


if __name__ == "__main__":
    main()
