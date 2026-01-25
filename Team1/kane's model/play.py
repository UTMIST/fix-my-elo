import argparse
import os
import time
from typing import Optional, List

import numpy as np
import torch
import chess
import chess.pgn

from .network import ChessNet
from .mcts import MCTS
from .board import move_to_index


"""
Run games with player or trained models.

Modes:
  1) human -> Human vs AI 
  2) aivai -> AI vs AI
  3) suggest -> Like human, but always prints AI suggestion before you make your move
"""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["human", "aivai", "suggest"], required=True,
                   help="human: play vs AI, aivai: AI vs AI, suggest: show AI suggestion then you enter move")
    p.add_argument("--sims", type=int, default=80, help="MCTS simulations per move")
    p.add_argument("--max_moves", type=int, default=300, help="Hard move cap")
    p.add_argument("--delay", type=float, default=0.0, help="Delay between moves (seconds)")
    p.add_argument("--pgn_out", type=str, required=True, help="PGN output path (required)")

    # Human/suggest mode
    p.add_argument("--model", type=str, default=None, help="Model path for AI in human/suggest mode")
    p.add_argument("--human_color", choices=["white", "black"], default="white", help="Human color in human/suggest mode")

    # AI vs AI mode
    p.add_argument("--white_model", type=str, default=None, help="Model path for White in aivai mode")
    p.add_argument("--black_model", type=str, default=None, help="Model path for Black in aivai mode")

    p.add_argument("--temp_moves", type=int, default=12, help="Number of early moves to sample (exploration). After this, play greedily.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_model(path: Optional[str], device: torch.device) -> ChessNet:
    model = ChessNet()
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def _safe_choice_from_policy(board: chess.Board, policy: np.ndarray) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves.")

    probs = []
    for mv in legal_moves:
        idx = move_to_index(mv, board)
        p = float(policy[idx]) if 0 <= idx < len(policy) else 0.0
        probs.append(p)

    probs_np = np.asarray(probs, dtype=np.float64)

    if not np.isfinite(probs_np).all():
        probs_np = np.ones(len(legal_moves), dtype=np.float64)

    probs_np = np.clip(probs_np, 0.0, None)
    s = float(probs_np.sum())
    if s <= 0.0:
        probs_np = np.ones(len(legal_moves), dtype=np.float64)
        s = float(probs_np.sum())

    probs_np /= s

    # Tiny drift fix
    drift = 1.0 - float(probs_np.sum())
    if abs(drift) > 1e-10:
        probs_np[-1] += drift
        if probs_np[-1] < 0.0:
            probs_np = np.clip(probs_np, 0.0, None)
            probs_np /= float(probs_np.sum())

    return np.random.choice(legal_moves, p=probs_np)


def select_ai_move(model: ChessNet, board: chess.Board, sims: int, move_count: int, temp_moves: int) -> chess.Move:
    mcts = MCTS(model, simulations=sims)
    policy = np.asarray(mcts.search(board), dtype=np.float32)

    s = float(policy.sum())
    if s > 0.0:
        policy /= s

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves.")

    if move_count < temp_moves:
        return _safe_choice_from_policy(board, policy)

    best_mv = None
    best_p = -1.0
    for mv in legal_moves:
        idx = move_to_index(mv, board)
        p = float(policy[idx]) if 0 <= idx < len(policy) else 0.0
        if p > best_p:
            best_p = p
            best_mv = mv

    return best_mv if best_mv is not None else legal_moves[0]


def parse_human_move(board: chess.Board, s: str) -> chess.Move:
    s = s.strip()
    try:
        mv = chess.Move.from_uci(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    try:
        mv = board.parse_san(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    raise ValueError("Invalid move. Use UCI (e2e4) or SAN (e4, Nf3, O-O).")


def save_pgn(moves: List[chess.Move], result: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    game = chess.pgn.Game()
    game.headers["Event"] = "chess_engine_kane"
    game.headers["Site"] = "local"
    game.headers["Result"] = result

    node = game
    for mv in moves:
        node = node.add_variation(mv)

    with open(out_path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")


def run_human_like(ai_model_path: str, human_color: str, sims: int, max_moves: int, temp_moves: int, delay: float,
                   device: torch.device, pgn_out: str, always_suggest: bool):
    board = chess.Board()
    ai = load_model(ai_model_path, device)
    moves: List[chess.Move] = []

    human_is_white = (human_color == "white")
    move_count = 0

    while (not board.is_game_over(claim_draw=True)) and move_count < max_moves:
        human_turn = (board.turn == chess.WHITE and human_is_white) or (board.turn == chess.BLACK and not human_is_white)

        if always_suggest:
            suggestion = select_ai_move(ai, board, sims=sims, move_count=move_count, temp_moves=temp_moves)
            print(f"AI suggests: {board.san(suggestion)} ({suggestion.uci()})")

        if human_turn:
            while True:
                s = input("Your move: ")
                try:
                    mv = parse_human_move(board, s)
                    break
                except Exception as e:
                    print(f"  {e}")
            board.push(mv)
            moves.append(mv)
        else:
            mv = select_ai_move(ai, board, sims=sims, move_count=move_count, temp_moves=temp_moves)
            print(f"AI plays: {board.san(mv)} ({mv.uci()})")
            board.push(mv)
            moves.append(mv)
            if delay > 0:
                time.sleep(delay)

        move_count += 1

    result = board.result(claim_draw=True)
    save_pgn(moves, result, pgn_out)
    print(f"PGN saved: {pgn_out}")


def run_ai_vs_ai(white_path: Optional[str], black_path: Optional[str], sims: int, max_moves: int, temp_moves: int,
                 delay: float, device: torch.device, pgn_out: str) -> None:
    board = chess.Board()
    white = load_model(white_path, device)
    black = load_model(black_path, device)
    moves: List[chess.Move] = []

    move_count = 0
    while (not board.is_game_over(claim_draw=True)) and move_count < max_moves:
        model = white if board.turn == chess.WHITE else black
        mv = select_ai_move(model, board, sims=sims, move_count=move_count, temp_moves=temp_moves)
        board.push(mv)
        moves.append(mv)
        move_count += 1
        if delay > 0:
            time.sleep(delay)

    result = board.result(claim_draw=True)
    save_pgn(moves, result, pgn_out)
    print(f"PGN saved: {pgn_out}")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("human", "suggest"):
        if not args.model:
            raise SystemExit("For --mode human/suggest you must provide --model PATH_TO_MODEL.pt")
        run_human_like(
            ai_model_path=args.model,
            human_color=args.human_color,
            sims=args.sims,
            max_moves=args.max_moves,
            temp_moves=args.temp_moves,
            delay=args.delay,
            device=device,
            pgn_out=args.pgn_out,
            always_suggest=(args.mode == "suggest")
        )
    else:
        run_ai_vs_ai(
            white_path=args.white_model,
            black_path=args.black_model,
            sims=args.sims,
            max_moves=args.max_moves,
            temp_moves=args.temp_moves,
            delay=args.delay,
            device=device,
            pgn_out=args.pgn_out
        )


if __name__ == "__main__":
    main()
