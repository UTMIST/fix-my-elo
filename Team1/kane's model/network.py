import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with two 3×3 convolutions."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolution + BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out = out + x

        # Final activation
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """Main neural network for policy and value prediction"""
    def __init__(self, num_blocks: int = 8,
                 channels: int = 96,
                 board_planes: int = 17,
                 num_moves: int = 4672):
        super().__init__()

        # Initial convolution
        self.conv_input = nn.Conv2d(board_planes, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ResidualBlock(channels)
            self.blocks.append(block)

        # Policy head layers
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_moves)

        # Value head layers
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass of the network."""
        
        # Input layer
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)

        # Residual blocks
        for block in self.blocks:
            out = block(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)

        value = torch.tanh(value)

        return policy, value
