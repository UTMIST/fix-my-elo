import argparse
import os

import numpy as np
import chess
import chess.pgn

from .board import board_to_tensor, move_to_index


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments for PGN processing.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            pgn (str, required): Path to the input PGN file.
            out (str, required): Path where the processed dataset (.npz) will be saved.
            max_games (int, optional): Maximum number of games to process. If None, all games are processed.
            min_elo (int, optional): Minimum average Elo required for a game to be included.
    """
    parser = argparse.ArgumentParser(description="Convert a PGN file into a training dataset.")
    parser.add_argument('--pgn', type=str, required=True,
                        help='Path to the PGN file containing games.')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to save the generated dataset.')
    parser.add_argument('--max_games', type=int, default=None,
                        help='Maximum number of games to process.')
    parser.add_argument('--min_elo', type=int, default=None,
                        help='Minimum average Elo of players to include a game.')
    return parser.parse_args()


def process_pgn(pgn_path: str, max_games: int = None, min_elo: int = None) -> tuple:
    """Convert PGN training files into training examples for learning.

    Each position in the game (before each move) is converted into a 17×8×8 tensor
    and the move is encoded as an integer index.

    """
    positions = []  # List for position tensors
    labels = []     # List for corresponding move indices
    games_processed = 0
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # No more games

            # Check if maximum number of games reached
            if max_games is not None and games_processed >= max_games:
                break

            # Elo filtering
            if min_elo is not None:
                white_elo = game.headers.get('WhiteElo')
                black_elo = game.headers.get('BlackElo')
                try:
                    w = int(white_elo)
                    b = int(black_elo)
                except (TypeError, ValueError):
                    continue
                avg_elo = (w + b) / 2
                if avg_elo < min_elo:
                    continue

            # Process all moves in the game
            board = game.board()
            for move in game.mainline_moves():
                # Record the position before the move
                tensor = board_to_tensor(board)
                positions.append(tensor)

                # Encode the move as an integer index
                # Pass the current board so that the encoder can mirror moves when Black is to move
                label = move_to_index(move, board)
                labels.append(label)
                # Make the move on the board
                board.push(move)
            games_processed += 1

    # Convert lists to NumPy arrays
    if positions:
        positions_array = np.stack(positions)
        labels_array = np.array(labels, dtype=np.int32)
    else:
        positions_array = np.empty((0, 17, 8, 8), dtype=np.float32)
        labels_array = np.empty((0,), dtype=np.int32)
    return positions_array, labels_array


def save_dataset(positions: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    """Save the positions and labels arrays to a compressed .npz file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, positions=positions, labels=labels)



    


if __name__ == '__main__':
    args = parse_arguments()
    print(f"Processing PGN file {args.pgn}")

    positions, labels = process_pgn(args.pgn, max_games=args.max_games, min_elo=args.min_elo)
    print(f"Parsed {len(labels)} positions from {args.pgn}.")
    print(f"Saving dataset to {args.out}")
    save_dataset(positions, labels, args.out)
    
    print("Done.")
