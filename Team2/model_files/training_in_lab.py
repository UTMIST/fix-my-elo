import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Team2.dataset import PGN_Dataset
from Team2.model_files.SLPolicyValueGPU import SLPolicyValueNetwork

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # MAX_GAMES = 1000000
    # MAX_VALIDATION_GAMES = 30000 # must be at least GAMES_PER_BATCH
    # GAMES_PER_BATCH = 30000
    MAX_GAMES = 10000
    MAX_VALIDATION_GAMES = 1000 # must be at least GAMES_PER_BATCH
    GAMES_PER_BATCH = 1000
    NUM_WORKERS = 20
    CHUNKSIZE = 1024
    model = SLPolicyValueNetwork().to(device)
    # model.load_state_dict(torch.load("sl_policy_network_KC.pth", map_location=torch.device("cpu")))
    policy_criterion = nn.CrossEntropyLoss()  # softmax regression loss function
    value_criterion = (
        nn.MSELoss()
    )  # use to use logistic loss but expects labels to be 0 or 1, not a range betwen -1 and 1
    optimizer = optim.Adam(model.parameters(), lr=0.1e-4)

    model.train()

    train_to_test_ratio = 0.9
    batch_size = 4096

    dataset = PGN_Dataset(
        "Team2/pgn_files/LumbrasGigaBase_OTB_ELO2400.pgn",
        max_games=MAX_GAMES,
        batchsize=GAMES_PER_BATCH,
    )
    dataset_generator = dataset.generate_dataset(num_workers=NUM_WORKERS, chunksize=CHUNKSIZE)
    # approximate number of batches to be allocated for validation
    # do this before so validation set stays the same
    num_validation_batches = math.ceil(
        dataset.length // dataset.batchsize * (1 - train_to_test_ratio)
    )
    # cap the validation set size
    if num_validation_batches * dataset.batchsize > MAX_VALIDATION_GAMES:
        num_validation_batches = MAX_VALIDATION_GAMES // GAMES_PER_BATCH
    # chunks the generator yields on a full pass, used for epoch progress
    total_chunks = math.ceil(dataset.length / dataset.batchsize)
    valid_boards = []
    valid_targets = []
    for _ in range(num_validation_batches):
        try:
            boards, targets = next(dataset_generator)
        except StopIteration:
            raise Exception(
                "your GAMES_PER_BATCH could be too large. Aim for 10:1 ratio for dataset.length:dataset.batchsize."
            )
        valid_boards.append(boards)
        valid_targets.append(targets)

    X_valid = torch.cat(valid_boards)
    t_valid = torch.cat(valid_targets)
    del valid_boards, valid_targets
    valid_dataset = TensorDataset(X_valid, t_valid)
    # num_workers=0: the dataset is already resident in this process, so worker
    # processes only add forks of a large parent and buy no I/O overlap
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # checkpoint = torch.load("checkpoint2.pth")
    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # start_epoch = checkpoint["epoch"] + 1
    # start_batch = checkpoint["batch"]
    start_epoch = 0
    start_batch = 0

    log_path = os.path.join(os.path.dirname(__file__), "lab_training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "avg_train_loss", "avg_val_loss", "epoch_time_s", "total_time_s"]
            )

    epochs = 100
    run_start = time.time()
    for epoch in range(epochs):

        epoch_start = time.time()
        batch_idx = 0
        chunk_idx = 0
        # every epoch skips the validation chunks, so they all see the same count
        chunks_this_epoch = total_chunks - num_validation_batches
        train_loss_sum = 0.0
        train_batches = 0
        # while there is another batch, pull it
        while True:
            try:
                X_train, t_train = next(dataset_generator)
            except StopIteration:
                # when out of batches, reset the generator and quit epoch.
                # skip past the validation chunks so they never get trained on
                dataset_generator = dataset.generate_dataset(
                    num_workers=NUM_WORKERS,
                    chunksize=CHUNKSIZE,
                    skip_chunks=num_validation_batches,
                )
                break
            # X_train is (N, 13, 8, 8) int8, t_train is (N, 2) int64 [move, winner]
            train_dataset = TensorDataset(X_train, t_train)
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            num_inner_batches = len(train_dataloader)
            chunk_idx += 1
            print(
                f"epoch {epoch + 1}/{epochs} progress: chunk {chunk_idx}/{chunks_this_epoch} "
                f"({chunk_idx / chunks_this_epoch * 100:.1f}%) "
                f"elapsed: {time.time() - epoch_start:.1f}s"
            )
            for index, (data, target) in enumerate(train_dataloader):
                # stored int8 to keep the chunk small, cast per minibatch on device
                data = data.to(device).float()
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = model(data)  # calculate predictions for this batch
                policy_loss = policy_criterion(
                    pred_policy, batch_move_target
                )  # calculate loss for policy
                value_loss = value_criterion(
                    pred_val, batch_val_target
                )  # calculate loss for value
                loss = policy_loss + value_loss
                optimizer.zero_grad()  # reset gradient
                loss.backward()  # calculate gradient
                optimizer.step()  # update parameters

                # print(
                #     f"epoch {epoch + 1} batch {index + 1}/{num_inner_batches} "
                #     f"({(index + 1) / num_inner_batches * 100:.1f}%) loss: {loss.item():.4f} "
                #     f"elapsed: {time.time() - epoch_start:.1f}s"
                # )
                train_loss_sum += loss.item()
                train_batches += 1
                batch_idx += 1

            # release this chunk before pulling the next, so peak memory stays
            # at one chunk rather than two
            del X_train, t_train, train_dataset, train_dataloader

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch": batch_idx,
            },
            "lab_trained.pth",
        )

        # check validation accuracy to see if general patterns are being learnt
        avg_train_loss = train_loss_sum / train_batches if train_batches else 0.0

        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_dataloader):
                data = data.to(device).float()
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = model(data)
                policy_loss = policy_criterion(
                    pred_policy, batch_move_target
                )  # calculate loss for policy
                value_loss = value_criterion(
                    pred_val, batch_val_target
                )  # calculate loss for value
                loss = policy_loss + value_loss
                # .item() so avg_val_loss is a float. accumulating the tensor
                # made the csv column read "tensor(10.8533)" instead of a number
                test_loss += loss.item()

        avg_val_loss = test_loss / len(valid_dataloader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start
        print(
            "epoch: {}, avg train loss: {:.6f}, avg valid loss: {:.6f}, epoch time: {:.1f}s, total time: {:.1f}s".format(
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                epoch_time,
                total_time,
            )
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    round(epoch_time, 1),
                    round(total_time, 1),
                ]
            )

        model.train()
