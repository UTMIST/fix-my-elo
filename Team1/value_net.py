import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, in_channels=18, channels=64, board_size=8):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.b1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.b2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.h1 = nn.Conv2d(channels, 32, 1)
        self.fc1 = nn.Linear(32 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def trunk(self, x):
        x = F.relu(self.stem(x))
        y = F.relu(self.b1(x))
        y = self.b2(y)
        return F.relu(x + y)

    def forward(self, x):
        t = self.trunk(x)
        v = F.relu(self.h1(t))
        v = v.flatten(1)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v)).squeeze(1)
        return v
