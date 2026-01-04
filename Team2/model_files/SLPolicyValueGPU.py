import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from data_processing import generate_dataset_from_pgn, label_to_move_table, fen_to_board_tensor
import chess
import random



class ResNetBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SLPolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=256, num_possible_moves=20480):
        super(SLPolicyValueNetwork, self).__init__()

        # shared trunk
        self.conv1 = nn.Conv2d(
            in_channels=13, out_channels=channels, kernel_size=3, padding=1 # padding=1 for same size output
        )

        self.norm = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(
            [ResNetBlock(channels) for _ in range(blocks)]
        )

        self.fc_shared = nn.Linear(channels * 8 * 8, 512)

        # policy head
        self.fc_policy = nn.Linear(512, num_possible_moves)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)


    def forward(self, x):
        # x = x.permute(0, 3, 1, 2) #conv2d expects (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        conv_out = self.norm(x)

        for block in self.blocks:
            conv_out = block(conv_out)

        # policy head
        policy_flat = torch.flatten(conv_out, start_dim=1)  # exclude batch dimension
        policy_features = F.relu(self.fc_shared(policy_flat))
        policy_logits = self.fc_policy(policy_features)

        # value head
        v = F.relu(self.value_bn(self.value_conv(conv_out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))


        return policy_logits, value
