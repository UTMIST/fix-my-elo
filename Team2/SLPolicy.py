import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SLPolicyNetwork(nn.Module):
    def __init__(self, num_possible_moves=4672):
        super(SLPolicyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_possible_moves)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

model = SLPolicyNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-3)

X = torch.randn((32, 8, 8, 12))  
y = torch.randint(0, 4672, (32,))

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")