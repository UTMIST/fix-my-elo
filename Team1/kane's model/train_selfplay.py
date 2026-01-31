import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import chess

from tqdm import tqdm

from .network import ChessNet
from .mcts import MCTS
from .board import board_to_tensor, move_to_index

ACTION_SIZE = 64 * 73  # AlphaZero-style 8x8x73 = 4672


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play reinforcement learning for chess AI training")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to initial model weights (.pt file).')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to save updated model weights')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of self-play games per iteration')
    parser.add_argument('--sims', type=int, default=50,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of self-play iterations')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    
    # Optional arguments for tensorboard logging and checkpointing
    parser.add_argument('--logdir', type=str, default=None,
                    help='TensorBoard log directory (e.g., runs/selfplay)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory to save checkpoints (e.g., checkpoints/)')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint (.pt) to resume from')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='Log to TensorBoard every N training positions')
    return parser.parse_args()


def _uniform_legal_policy(board: chess.Board) -> np.ndarray:
    """Fallback policy: uniform over legal moves in ACTION_SIZE action space."""
    p = np.zeros((ACTION_SIZE,), dtype=np.float32)
    legal = list(board.legal_moves)
    if not legal:
        return p
    for mv in legal:
        idx = move_to_index(mv, board)
        if 0 <= idx < ACTION_SIZE:
            p[idx] = 1.0
    s = float(np.sum(p))
    if s > 0.0:
        p /= s
    return p


def _safe_choice_from_policy(board: chess.Board, policy_np: np.ndarray) -> chess.Move:
    """
    Choose a legal move using policy probabilities
    - mask legal moves
    - replace NaN/Inf with 0
    - clamp negatives
    - renormalize
    - fallback to uniform if needed
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves available to sample from.")

    move_probs = np.zeros(len(legal_moves), dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        idx = move_to_index(mv, board)
        if 0 <= idx < policy_np.shape[0]:
            move_probs[i] = float(policy_np[idx])
        else:
            move_probs[i] = 0.0

    # Replace NaN/Inf with 0
    move_probs[~np.isfinite(move_probs)] = 0.0
    # Clamp negatives to 0
    move_probs = np.clip(move_probs, 0.0, None)

    s = float(move_probs.sum())
    if s <= 0.0:
        # Uniform fallback
        move_probs[:] = 1.0 / float(len(legal_moves))
    else:
        move_probs /= s

    # At this point: probs are finite, non-negative, sum ~ 1
    return np.random.choice(legal_moves, p=move_probs)


def play_game(network: ChessNet, mcts_simulations: int) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Play one self-play game using MCTS, recording (state, pi, z)."""
    board = chess.Board()
    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []

    root_player = board.turn
    mcts_engine = MCTS(network, simulations=mcts_simulations)

    move_count = 0
    max_moves = 200  # hard cap to avoid infinite games, can adjust

    move_pbar = tqdm(total=max_moves, desc="  Playing game (moves)", unit="move", leave=False)

    while (not board.is_game_over(claim_draw=True)) and (move_count < max_moves):
        state_tensor = board_to_tensor(board)
        states.append(state_tensor)

        policy = mcts_engine.search(board)  # should be length ACTION_SIZE = 4672
        policy_np = np.asarray(policy, dtype=np.float32)

        # ensure correct length
        if policy_np.shape[0] != ACTION_SIZE:
            fixed = np.zeros((ACTION_SIZE,), dtype=np.float32)
            n = min(ACTION_SIZE, policy_np.shape[0])
            fixed[:n] = policy_np[:n]
            policy_np = fixed

        # Replace NaN/Inf with 0
        policy_np[~np.isfinite(policy_np)] = 0.0
        # Clamp negatives
        policy_np = np.clip(policy_np, 0.0, None)

        s = float(np.sum(policy_np))
        if s > 0.0:
            policy_np /= s
        else:
            policy_np = _uniform_legal_policy(board)

        policies.append(policy_np)

        selected_move = _safe_choice_from_policy(board, policy_np)
        board.push(selected_move)

        move_count += 1
        move_pbar.update(1)

    move_pbar.close()

    # Outcome (z) from root player's perspective
    if board.is_checkmate():
        winner = not board.turn
        outcome = 1.0 if winner == root_player else -1.0
    else:
        # includes all draws and max_moves cutoff
        outcome = 0.0

    return states, policies, outcome


def train_on_selfplay_data(model: ChessNet,
                           games_data: List[Tuple[List[np.ndarray], List[np.ndarray], float]],
                           optimizer: torch.optim.Optimizer,
                           device: torch.device) -> float:
    """Train model on self-play data; returns avg loss per position."""
    model.train()
    total_loss = 0.0
    total_positions = 0
    l2_coeff = 1e-4

    total_positions_expected = sum(len(gs) for (gs, _gp, _gr) in games_data)

    # tqdm progress bar over positions
    pbar = tqdm(total=total_positions_expected, desc="  Training on self-play", unit="pos", leave=False)

    for game_states, game_policies, game_result in games_data:
        for i in range(len(game_states)):
            state_np = game_states[i]
            policy_np = game_policies[i]

            # enforce ACTION_SIZE
            if policy_np.shape[0] != ACTION_SIZE:
                fixed = np.zeros((ACTION_SIZE,), dtype=np.float32)
                n = min(ACTION_SIZE, policy_np.shape[0])
                fixed[:n] = policy_np[:n]
                policy_np = fixed

            state_tensor = torch.from_numpy(state_np).unsqueeze(0).float().to(device)
            policy_tensor = torch.from_numpy(policy_np).unsqueeze(0).float().to(device)
            target_value = torch.tensor([game_result], dtype=torch.float32, device=device)

            logits, value = model(state_tensor)

            # Safety: if logits shape differs, stop with a clear error
            if logits.shape[1] != ACTION_SIZE:
                raise RuntimeError(
                    f"Network policy head output is {logits.shape[1]}, but ACTION_SIZE is {ACTION_SIZE}. "
                    f"Fix network.py to output ACTION_SIZE."
                )

            log_probs = F.log_softmax(logits, dim=1)
            # Policy loss: cross-entropy with soft targets (MCTS pi)
            policy_loss = -(policy_tensor * log_probs).sum(dim=1).mean()

            # Value loss: MSE between predicted value and game outcome
            value_loss = F.mse_loss(value.view(-1), target_value.view(-1))

            # L2 regularization
            l2_loss = 0.0
            for param in model.parameters():
                l2_loss = l2_loss + torch.sum(param * param)

            loss = policy_loss + value_loss + l2_coeff * l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_positions += 1

            avg_so_far = total_loss / max(1, total_positions)
            pbar.set_postfix(loss=f"{avg_so_far:.4f}")
            pbar.update(1)

    pbar.close()
    return (total_loss / float(total_positions)) if total_positions > 0 else 0.0


def main():
    args = parse_arguments()
    model = ChessNet()

    if args.model is not None and os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.model}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for iteration in range(args.iterations):
        games_data: List[Tuple[List[np.ndarray], List[np.ndarray], float]] = []

        game_bar = tqdm(range(args.games), desc=f"Iteration {iteration + 1}/{args.iterations} (games)", unit="game")
        for _ in game_bar:
            states, policies, outcome = play_game(model, args.sims)
            games_data.append((states, policies, outcome))
            game_bar.set_postfix(last_result=outcome)

        avg_loss = train_on_selfplay_data(model, games_data, optimizer, device)
        print(f"Iteration {iteration + 1}: average loss = {avg_loss:.4f}")

        # Save after each iteration
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"[checkpoint] Saved -> {args.out}")

    torch.save(model.state_dict(), args.out)
    print(f"Updated model saved to {args.out}")


if __name__ == '__main__':
    main()
