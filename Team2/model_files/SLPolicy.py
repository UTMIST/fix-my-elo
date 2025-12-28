import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from data_processing import generate_dataset_from_pgn, label_to_move_table, fen_to_board
import chess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = generate_dataset_from_pgn("team2/masaurus101-white.pgn")
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
train_dataset = TensorDataset(X_train, t_train)
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


model = SLPolicyNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)


def predict_move(model, board_tensor):
    """
    Takes a board tensor (8, 8, 12) and returns the predicted UCI move.
    """
    label_to_uci = label_to_move_table()
    model.eval()  # Set to evaluation mode

    with torch.no_grad():  # No gradients needed for inference
        # Add batch dimension: (8, 8, 12) -> (1, 8, 8, 12)
        board_batch = board_tensor.unsqueeze(0)

        # Get model output
        logits = model(board_batch)  # Shape: (1, 20480)
        probabilities = F.softmax(logits, dim=1)

        # Get the highest scoring move
        predicted_label = torch.argmax(logits, dim=1).item()

        # Convert to UCI
        predicted_uci = label_to_uci[predicted_label]

    return predicted_uci, predicted_label, probabilities[0][predicted_label]


def list_predicted_moves(model, board_tensor, num_moves):
    label_to_uci = label_to_move_table()

    model.eval()
    with torch.no_grad():
        board_batch = board_tensor.unsqueeze(0)
        logits = model(board_batch)
        probabilities = F.softmax(logits, dim=1)
        score, moves = torch.topk(probabilities, num_moves)
        moves = [label_to_uci[int(move)] for move in moves[0]]

    return moves, score


for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        output = model(data)  # calculate predictions for this batch
        loss = criterion(output, target)  # calculate loss
        optimizer.zero_grad()  # reset gradient
        loss.backward()  # calculate gradient
        optimizer.step()  # update parameters

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).type(torch.float).sum().item()

    print(
        "epoch: {}, test loss: {:.6f}, test accuracy: {:.6f}".format(
            epoch + 1,
            test_loss / len(test_dataloader),
            correct / len(test_dataloader.dataset),
        )
    )


board = chess.Board()
board_tensor = fen_to_board(board.fen())
predict_move(model, board_tensor)
print(list_predicted_moves(model, board_tensor, num_moves=5))
