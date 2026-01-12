import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from data_processing import generate_dataset_from_pgn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dataset = generate_dataset_from_pgn("../pgn_files/LumbrasGigaBase_OTB_2025.pgn", max_games=100000)

train_to_test_ratio = 0.8

train_size = int(len(dataset) * train_to_test_ratio)
test_size = len(dataset) - train_size

# split the dataset
train_data = dataset[:train_size]
test_data = dataset[train_size:]

X_train = torch.stack([board for board, move, winner in train_data])  # (N, 8, 8, 12)
t_train = torch.tensor([(move, winner) for board, move, winner in train_data])  # (N, 2)

X_test = torch.stack([board for board, move, winner in test_data])
t_test = torch.tensor([(move, winner) for board, move, winner in test_data])

# create DataLoaders
batch_size = 3048
train_dataset = TensorDataset(X_train, t_train)
test_dataset = TensorDataset(X_test, t_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


model = SLPolicyValueNetwork().to(device)
# model.load_state_dict(torch.load("sl_policy_network_KC.pth", map_location=torch.device("cpu")))
policy_criterion = nn.CrossEntropyLoss() # softmax regression loss function
value_criterion = nn.MSELoss() # use to use logistic loss but expects labels to be 0 or 1, not a range betwen -1 and 1
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)

model.train()

# checkpoint = torch.load("checkpoint2.pth")
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# start_epoch = checkpoint["epoch"] + 1
# start_batch = checkpoint["batch"]
start_epoch = 0
start_batch = 0

print(f"Resuming from epoch {start_epoch}, batch {start_batch}")

epochs = 100

for epoch in range(epochs):

    epoch = start_epoch + epoch
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        batch_move_target = target[:, 0].to(device)
        batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

        pred_policy, pred_val = model(data)  # calculate predictions for this batch
        policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
        value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
        loss = policy_loss + value_loss
        optimizer.zero_grad()  # reset gradient
        loss.backward()  # calculate gradient
        optimizer.step()  # update parameters

        if batch_idx % 100 == 0:
            print(f"batch progress: epoch {epoch} {(100 *(batch_idx +1)/len(train_dataloader)):.2f}% loss: {loss.item():.6f}")

        print(f"batch progress: epoch {epoch} {(100 *(batch_idx +1)/len(train_dataloader)):.2f}% loss: {loss.item():.6f}")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, 
        "batch": batch_idx,
    }, "lab_trained.pth")
    print('model checkpoint saved')

    # check validation accuracy to see if general patterns are being learnt

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data = data.to(device)
            batch_move_target = target[:, 0].to(device)
            batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

            pred_policy, pred_val = model(data)
            policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
            value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
            loss = policy_loss + value_loss
            test_loss += loss

    print('epoch: {}, test loss: {:.6f}'.format(
        epoch + 1,
        test_loss / len(test_dataloader),
        ))
    
    model.train()
