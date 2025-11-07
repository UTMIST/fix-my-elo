import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from encoders import MoveEncoder, ChessBoardEncoder
from torch.utils.data import Dataset, DataLoader
from typing import List
from policy_net import PolicyNet


class ChessDataset(Dataset):
    """Custom Dataset for Chess Positions and Moves."""

    def __init__(self, positions: List[chess.Board], moves: List[chess.Move]):
        self.positions = positions
        self.moves = moves
        self.move_encoder = MoveEncoder()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        board = self.positions[index]
        move = self.moves[index]

        board_tensor = ChessBoardEncoder.board_to_tensor(board)
        move_index = self.move_encoder.move_to_index(move)

        return board_tensor, torch.tensor(move_index, dtype=torch.long)

def train_policy_net(model: PolicyNet, dataset: ChessDataset, epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train the Policy Network."""
    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        batches = 0

        for batch_index, (boards, moves) in enumerate(loader):
            boards = boards.to(device)
            moves = moves.to(device)

            optimizer.zero_grad()
            probs = model(boards)

            loss = criterion(probs, moves)

            loss.backward()
            optimizer.step()

            total_loss =+ loss.item()
            batches += 1

            if batch_index % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_index}, Loss: {loss.item()}")

        avg_loss = total_loss / batches
        print(f'Epoch [{epoch + 1}/{epochs}] completed. Average Loss: {avg_loss}')

if __name__ == "__main__":
    model = PolicyNet(num_filters=128)  # Smaller for demo
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    sample_positions = []
    sample_moves = []
    # sample positions
    board = chess.Board()
    for _ in range(1000):
        if not board.is_game_over() and len(list(board.legal_moves)) > 0:
            legal_moves = list(board.legal_moves)
            move = legal_moves[0]
            # uses first legal move, should be replaced with MCTS output or actual dataset

            sample_positions.append(board.copy())
            sample_moves.append(move)
            board.push(move)
        else:
            board = chess.Board()

    dataset = ChessDataset(sample_positions, sample_moves)
    print(f"Dataset size: {len(dataset)}")

    train_policy_net(
        model,
        dataset,
        epochs=2,
        batch_size=64,
        learning_rate=0.001
    )

    # 5. Make a prediction
    test_board = chess.Board()
    predicted_move = model.predict(test_board)
    print(f"Predicted move: {predicted_move}")