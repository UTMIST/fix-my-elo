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
    print(f"[load] dataset ready: {dataset.length} games, batchsize {dataset.batchsize}")
    dataset_generator = dataset.generate_dataset(num_workers=NUM_WORKERS, chunksize=CHUNKSIZE)
    print(f"[load] generator started: {NUM_WORKERS} workers, chunksize {CHUNKSIZE}")
    # approximate number of batches to be allocated for validation
    # do this before so validation set stays the same
    num_validation_batches = math.ceil(
        dataset.length // dataset.batchsize * (1 - train_to_test_ratio)
    )
    # cap the validation set size
    if num_validation_batches * dataset.batchsize > MAX_VALIDATION_GAMES:
        num_validation_batches = MAX_VALIDATION_GAMES // GAMES_PER_BATCH
    print(f"using {num_validation_batches} batches ({num_validation_batches*dataset.batchsize} games) for validation")
    
    # chunks the generator yields on a full pass, used for epoch progress
    total_chunks = math.ceil(dataset.length / dataset.batchsize)
    valid_boards = []
    valid_targets = []
    for i in range(num_validation_batches):
        try:
            boards, targets = next(dataset_generator)
        except StopIteration:
            raise Exception(
                "your GAMES_PER_BATCH could be too large. Aim for 10:1 ratio for dataset.length:dataset.batchsize."
            )
        print(f"[load] valid chunk {i + 1}/{num_validation_batches}: {boards.shape[0]} positions")
        valid_boards.append(boards)
        valid_targets.append(targets)

    X_valid = torch.cat(valid_boards)
    t_valid = torch.cat(valid_targets)
    print(f"[load] valid set built: {X_valid.shape[0]} positions")
    del valid_boards, valid_targets
    valid_dataset = TensorDataset(X_valid, t_valid)
    # num_workers=0: the dataset is already resident in this process, so worker
    # processes only add forks of a large parent and buy no I/O overlap
    # drop_last here too: validation is a second allocation regime (no_grad, no
    # gradient buffers) with its own odd trailing batch. it also fixes the
    # averaging: avg loss divides by batch count, so an unequal last batch would
    # be weighted the same as a full one
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
    )
    print(f"[load] valid dataloader ready: {len(valid_dataloader)} minibatches")

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
                [
                    "epoch",
                    "avg_train_policy_loss",
                    "avg_train_value_loss",
                    "avg_train_loss",
                    "avg_val_policy_loss",
                    "avg_val_value_loss",
                    "avg_val_loss",
                    "epoch_time_s",
                    "total_time_s",
                ]
            )

    epochs = 100
    run_start = time.time()
    checkpoint_path = "lab_trained.pth"
    # cumulative over the whole run, so a checkpoint says how much data the
    # weights in it have actually seen
    games_seen = 0
    positions_seen = 0
    chunks_seen = 0
    for epoch in range(epochs):

        epoch_start = time.time()
        batch_idx = 0
        chunk_idx = 0
        # every epoch skips the validation chunks, so they all see the same count
        chunks_this_epoch = total_chunks - num_validation_batches
        train_loss_sum = 0.0
        train_policy_sum = 0.0
        train_value_sum = 0.0
        train_batches = 0
        # split the epoch into time spent waiting on data vs time in the model
        data_time_sum = 0.0
        compute_time_sum = 0.0
        # while there is another batch, pull it
        while True:
            fetch_start = time.time()
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
                print(f"[load] generator reset, skipping {num_validation_batches} valid chunks")
                break
            fetch_time = time.time() - fetch_start
            data_time_sum += fetch_time
            print(
                f"[load] train chunk {chunk_idx + 1}: {X_train.shape[0]:,} positions, "
                f"waited {fetch_time:.1f}s on the generator"
            )
            # X_train is (N, 13, 8, 8) int8, t_train is (N, 2) int64 [move, winner]
            build_start = time.time()
            train_dataset = TensorDataset(X_train, t_train)
            # drop_last: the trailing partial batch is a different odd size every
            # chunk, and each distinct size strands an unusable remnant in the cuda
            # allocator. dropping it makes every allocation identical so blocks
            # recycle exactly. costs <0.4% of the chunk, and shuffle reshuffles the
            # dropped tail each pass so no data is permanently excluded
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            num_inner_batches = len(train_dataloader)
            print(
                f"[load] train dataloader ready: {num_inner_batches} minibatches, "
                f"built in {time.time() - build_start:.3f}s"
            )
            chunk_idx += 1
            print(
                f"epoch {epoch + 1}/{epochs} progress: chunk {chunk_idx}/{chunks_this_epoch} "
                f"({chunk_idx / chunks_this_epoch * 100:.1f}%) "
                f"elapsed: {time.time() - epoch_start:.1f}s"
            )
            compute_start = time.time()
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

                # one .item() per component, then add on the host. loss.item()
                # would be a third gpu sync for a number we already have
                policy_l = policy_loss.item()
                value_l = value_loss.item()
                total_l = policy_l + value_l
                print(
                    f"      epoch {epoch + 1} batch {index + 1}/{num_inner_batches} "
                    f"({(index + 1) / num_inner_batches * 100:.1f}%) "
                    f"policy: {policy_l:.4f} value: {value_l:.4f} total: {total_l:.4f} "
                    f"elapsed: {time.time() - epoch_start:.1f}s"
                )
                train_policy_sum += policy_l
                train_value_sum += value_l
                train_loss_sum += total_l
                train_batches += 1
                batch_idx += 1

            compute_time = time.time() - compute_start
            compute_time_sum += compute_time
            chunk_total = fetch_time + compute_time
            print(
                f"[time] chunk {chunk_idx}: data {fetch_time:6.1f}s | "
                f"compute {compute_time:6.1f}s | "
                f"{fetch_time / max(chunk_total, 1e-9) * 100:.0f}% of the chunk was spent waiting on data"
            )

            chunks_seen += 1
            games_seen += GAMES_PER_BATCH
            positions_seen += X_train.shape[0]

            # checkpoint every chunk, not every epoch, so a crash costs one
            # chunk of work instead of a whole epoch. write to a temp file and
            # rename: os.replace is atomic on posix, so a crash mid-write
            # leaves the previous good checkpoint intact rather than a
            # truncated file
            save_start = time.time()
            tmp_path = checkpoint_path + ".tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # exactly where this was taken
                    "epoch": epoch,  # 0-indexed, epoch currently in progress
                    "chunk": chunk_idx,  # 1-based, chunks finished this epoch
                    "batch": batch_idx,  # minibatches finished this epoch
                    "chunks_this_epoch": chunks_this_epoch,
                    # enough to reconstruct how much data these weights saw
                    "games_per_batch": GAMES_PER_BATCH,
                    "max_games": MAX_GAMES,
                    "num_validation_batches": num_validation_batches,
                    "chunks_seen": chunks_seen,
                    "games_seen": games_seen,
                    "positions_seen": positions_seen,
                },
                tmp_path,
            )
            os.replace(tmp_path, checkpoint_path)
            print(
                f"[ckpt] epoch {epoch + 1} chunk {chunk_idx}/{chunks_this_epoch} "
                f"batch {batch_idx} | {games_seen:,} games / {positions_seen:,} positions "
                f"seen | saved in {time.time() - save_start:.1f}s"
            )

            # release this chunk before pulling the next, so peak memory stays
            # at one chunk rather than two
            del X_train, t_train, train_dataset, train_dataloader

        # no epoch-level save: the last chunk of the epoch already wrote one,
        # and validation does not change the weights

        # check validation accuracy to see if general patterns are being learnt
        avg_train_policy = train_policy_sum / train_batches if train_batches else 0.0
        avg_train_value = train_value_sum / train_batches if train_batches else 0.0
        avg_train_loss = train_loss_sum / train_batches if train_batches else 0.0

        model.eval()
        test_loss = 0
        test_policy_sum = 0.0
        test_value_sum = 0.0
        correct = 0
        valid_start = time.time()

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
                # .item() so the averages are floats. accumulating the tensor
                # made the csv column read "tensor(10.8533)" instead of a number
                policy_l = policy_loss.item()
                value_l = value_loss.item()
                test_policy_sum += policy_l
                test_value_sum += value_l
                test_loss += policy_l + value_l

        valid_time = time.time() - valid_start
        num_valid_batches = len(valid_dataloader)
        avg_val_policy = test_policy_sum / num_valid_batches
        avg_val_value = test_value_sum / num_valid_batches
        avg_val_loss = test_loss / num_valid_batches
        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start
        print(
            "epoch: {}, epoch time: {:.1f}s, total time: {:.1f}s".format(
                epoch + 1, epoch_time, total_time
            )
        )
        print(
            "  train  policy {:.6f} | value {:.6f} | total {:.6f}".format(
                avg_train_policy, avg_train_value, avg_train_loss
            )
        )
        print(
            "  valid  policy {:.6f} | value {:.6f} | total {:.6f}".format(
                avg_val_policy, avg_val_value, avg_val_loss
            )
        )
        # where the epoch actually went. "other" is checkpoint save plus the
        # final next() that raises StopIteration and rebuilds the generator
        other = epoch_time - data_time_sum - compute_time_sum - valid_time
        print(
            "  breakdown: data {:.1f}s ({:.0f}%) | compute {:.1f}s ({:.0f}%) | "
            "valid {:.1f}s ({:.0f}%) | other {:.1f}s".format(
                data_time_sum, data_time_sum / max(epoch_time, 1e-9) * 100,
                compute_time_sum, compute_time_sum / max(epoch_time, 1e-9) * 100,
                valid_time, valid_time / max(epoch_time, 1e-9) * 100,
                other,
            )
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    float(avg_train_policy),
                    float(avg_train_value),
                    float(avg_train_loss),
                    float(avg_val_policy),
                    float(avg_val_value),
                    float(avg_val_loss),
                    float(round(epoch_time, 1)),
                    float(round(total_time, 1)),
                ]
            )

        model.train()
