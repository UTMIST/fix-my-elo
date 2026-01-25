import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from .network import ChessNet


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chess network using supervised learning")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset (.npz) created by data_preparation.py')
    parser.add_argument('--model_out', type=str, required=True,
                        help='Path to save the trained model weights (.pt)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for Adam optimiser (default: 1e-3)')
    return parser.parse_args()


def load_dataset(path: str) -> tuple:
    with np.load(path) as data:
        positions = data['positions']
        labels = data['labels']
    return positions, labels


def train_supervised(model: ChessNet, positions: np.ndarray, labels: np.ndarray,
                     epochs: int, batch_size: int, lr: float, device: torch.device,
                     out_path: str):
    positions_tensor = torch.from_numpy(positions).float()
    labels_tensor = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(positions_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # show dataset size
    print(f"Loaded dataset: {len(dataset)} positions | batch_size={batch_size} | device={device}")

    for epoch in range(epochs):
        total_loss = 0.0

        # tqdm progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=True)
        for pos_batch, label_batch in pbar:
            pos_batch = pos_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits, _value = model(pos_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            avg_so_far = total_loss / max(1, (pbar.n + 1))
            pbar.set_postfix(loss=f"{avg_so_far:.4f}")

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained model to {out_path}")


def main() -> None:
    args = parse_arguments()
    if not os.path.exists(args.data):
        print(f"Error: dataset file '{args.data}' does not exist.")
        return

    positions, labels = load_dataset(args.data)
    model = ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_supervised(model, positions, labels, args.epochs, args.batch_size,
                     args.lr, device, args.model_out)


if __name__ == '__main__':
    main()
