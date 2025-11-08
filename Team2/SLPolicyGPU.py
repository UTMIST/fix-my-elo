import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from torch.utils.data import DataLoader, TensorDataset
from data_processing import generate_dataset_from_pgn, label_to_move_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = generate_dataset_from_pgn("Team2/masaurus101-white.pgn") # dataset is a list of all moves in a game (8,8,12)
train_to_test_ratio = 0.8

train_size = int(len(dataset) * train_to_test_ratio)
test_size = len(dataset) - train_size

# Split the dataset
train_data = dataset[:train_size]
test_data = dataset[train_size:]

# Convert to tensors (simpler now since labels are already integers!)
X_train = torch.stack([board for board, label in train_data])  # (N, 8, 8, 12)
t_train = torch.tensor([label for board, label in train_data])  # (N,)

X_test = torch.stack([board for board, label in test_data])
t_test = torch.tensor([label for board, label in test_data])

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train, t_train) # pairs x and y
test_dataset = TensorDataset(X_test, t_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SLPolicyNetwork(nn.Module):
    def __init__(self, num_possible_moves=20480):
        super(SLPolicyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=12, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_possible_moves)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)  # exclude batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = SLPolicyNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-3)

label_to_uci = label_to_move_table()

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def predict_move(model, board_tensor):
    """
    Takes a board tensor (8, 8, 12) and returns the predicted UCI move.
    """
    model.eval()  # Set to evaluation mode

    with torch.no_grad():  # No gradients needed for inference
        # Add batch dimension: (8, 8, 12) -> (1, 8, 8, 12)
        board_batch = board_tensor.unsqueeze(0).to(device)

        # Get model output
        outputs = model(board_batch)  # Shape: (1, 20480)

        # Get the highest scoring move
        predicted_label = torch.argmax(outputs, dim=1).item()

        # Convert to UCI
        predicted_uci = label_to_uci[predicted_label]

    return predicted_uci, predicted_label


# Example usage:
# Get a board from your dataset
board_tensor, actual_label = dataset[0]

# Predict
predicted_uci, predicted_label = predict_move(model, board_tensor)
actual_uci = label_to_uci[actual_label]

print(f"Predicted move: {predicted_uci}")
print(f"Actual move: {actual_uci}")
print(f"Match: {predicted_uci == actual_uci}")