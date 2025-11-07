import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from encoders import MoveEncoder, ChessBoardEncoder


class ResBlock(nn.Module):
    """Residual Block with two convolutional layers."""

    def __init__(self, num_filters: int = 256):
        super(ResBlock, self).__init__()

        self.convl1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.convl2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x

        out = self.convl1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.convl2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out

class PolicyNet(nn.Module):
    """Policy Network for Chess Move Prediction."""
    # more res blocks and filters can be added for better performance
    def __init__(self, num_res_blocks: int = 5, num_filters: int = 256):
        super(PolicyNet, self).__init__()

        self.num_filters = num_filters
        self.move_encoder = MoveEncoder()

        #Input layer
        self.input_convl = nn.Conv2d(14, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual Block Tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_convl = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, self.move_encoder.num_moves)

    def forward(self, x):
        x = self.input_convl(x)
        x = self.input_bn(x)
        x = F.relu(x)

        for residual in self.res_blocks:
            x = residual(x)

        policy = self.policy_convl(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)

        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # To turn CNN output into probabilities
        policy = F.log_softmax(policy, dim=1)

        return policy

    def predict(self, board: chess.Board) -> np.ndarray:
        """Predict move probabilities for a given chess board state.
        """
        self.eval()
        with torch.no_grad():
            board_tensor = ChessBoardEncoder.board_to_tensor(board)
            board_tensor = board_tensor.unsqueeze(0)

            probs = self.forward(board_tensor)
            probs = torch.exp(probs).squeeze(0)

            legal_moves = list(board.legal_moves)
            best_move = None
            best_prob = -1

            for move in legal_moves:
                move_index = self.move_encoder.move_to_index(move)
                if move_index >= 0 and probs[move_index] > best_prob:
                    best_prob = probs[move_index]
                    best_move = move

            return best_move if best_move else legal_moves[0]