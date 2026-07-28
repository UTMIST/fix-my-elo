import gc
import os
from Team2.model_files.deprecated.data_processing import (
    fen_to_board_tensor,
    uci_to_tensor,
    move_tensor_to_label,
    extract_from_fen,
    extract_from_fen_without_multiprocessing,
)
import chess.pgn
import time
import torch


def worker(data: tuple[str, str, str]):
    """datapoint: list[list[FEN, move_played, winner]]"""

    fen = data[0]
    move = data[1]
    winner = data[2]

    board = fen_to_board_tensor(fen)
    move_tensor = uci_to_tensor(move)
    move = move_tensor_to_label(move_tensor)

    return (board.numpy(), move, winner)


def game_to_datapoint(game: chess.pgn.Game):
    board = game.board()
    datapoints = []

    cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
    winner = cases[game.headers["Result"]]

    current_player_eval = (
        1  # flip evaluation to reflect whether the CURRENT player eventually won or not
    )
    # we want to know if the current player won, as position eval depends on whose turn it is
    for move in game.mainline_moves():
        datapoints.append([board.fen(), move.uci(), current_player_eval * winner])
        board.push(move)
        current_player_eval *= -1
    return datapoints


class PGN_Dataset:
    def __init__(self, path, max_games, batchsize):
        # it is advised to set max_games:batchsize to be around 10:1
        self.path = path
        self.max_games = max_games
        self.batchsize = batchsize

        # SkipVisitor tokenises each game without building a board or parsing
        # SAN, ~135x faster than read_game and counts identically. a game with
        # invalid SAN is counted by both: read_game records the error and
        # truncates the mainline, it never drops the game
        # flush on every progress line: stdout is block-buffered when it is
        # redirected to a file, so without this nothing appears for minutes
        size_mb = os.path.getsize(path) / 1e6
        print(
            f"[scan] counting games in {os.path.basename(path)} ({size_mb:,.0f} MB), "
            f"capped at {max_games:,}...",
            flush=True,
        )
        scan_start = time.perf_counter()
        count = 0
        with open(path) as f:
            while count < max_games and chess.pgn.skip_game(f):
                count += 1
        self.length = count
        scan_time = time.perf_counter() - scan_start
        print(
            f"[scan] counted {self.length:,} games in {scan_time:.1f}s "
            f"({self.length / max(scan_time, 1e-9):,.0f} games/s, skip-parse only)",
            flush=True,
        )

    def generate_dataset(self, num_workers, chunksize, skip_chunks=0):
        # while count <= max_games
        #   load games_per_batch games from path using chess.pgn.readGame
        #   process them all in parallel with helpers to return a batch
        #   yield that batch
        # skip_chunks drops the first N chunks of the pass, so callers can keep
        # a held-out split (the validation chunks) out of training on every
        # reset. reads are sequential and imap preserves order, so chunk k is
        # the same games on every pass. skipped games still count toward
        # max_games, which keeps the chunk boundaries identical to an unskipped
        # pass
        skipped_games = min(skip_chunks * self.batchsize, self.max_games)
        count = skipped_games
        with open(self.path) as f:
            skip_start = time.perf_counter()
            for _ in range(skipped_games):
                # SkipVisitor walks the game without parsing moves, so this
                # costs a file scan rather than a board replay
                if not chess.pgn.skip_game(f):
                    break
            skip_time = time.perf_counter() - skip_start
            if skipped_games:
                print(
                    f"[pass] skipped {skipped_games:,} held-out games in {skip_time:.1f}s"
                )
            # totals for the whole pass, so the serial share is visible
            pass_read = 0.0
            pass_extract = 0.0
            pass_positions = 0
            batch = []
            bad_games = 0
            # split the chunk cost into its two phases: pgn parsing in this
            # process, then tensor extraction across the worker pool
            read_start = time.perf_counter()
            chunk_first_game = count
            last_report = read_start
            print(
                f"  reading games {count + 1:,}-{count + self.batchsize:,} "
                f"(serial, single core)...",
                flush=True,
            )
            while count <= self.max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                # read_game does not drop a game with invalid SAN, it records
                # the error and truncates the mainline there, so those games
                # contribute fewer positions. count them so the loss is visible
                if game.errors:
                    bad_games += 1
                # process this game
                batch += game_to_datapoint(game)

                count += 1

                # heartbeat on a time interval, not a game interval, so the
                # line count stays bounded whatever batchsize is set to
                now = time.perf_counter()
                if now - last_report >= 30:
                    done = count - chunk_first_game
                    rate = done / max(now - read_start, 1e-9)
                    print(
                        f"    ...{done:,}/{self.batchsize:,} games read "
                        f"({now - read_start:.0f}s elapsed, {rate:,.0f} games/s, "
                        f"~{(self.batchsize - done) / max(rate, 1e-9):.0f}s left)",
                        flush=True,
                    )
                    last_report = now

                if count % self.batchsize == 0 or count == self.max_games:
                    read_time = time.perf_counter() - read_start
                    # yields (boards int8 (N,13,8,8), targets int64 (N,2)).
                    extract_start = time.perf_counter()
                    boards, targets = extract_from_fen(
                        batch, num_workers=num_workers, chunksize=chunksize
                    )
                    extract_time = time.perf_counter() - extract_start

                    positions = len(batch)
                    pass_read += read_time
                    pass_extract += extract_time
                    pass_positions += positions
                    print(
                        f"  chunk to game {count}: {positions:,} positions | "
                        f"pgn read {read_time:6.1f}s (pass total {pass_read:7.1f}s) | "
                        f"extract {extract_time:6.1f}s "
                        f"({positions/max(extract_time, 1e-9):,.0f} pos/s) | "
                        f"total {read_time + extract_time:6.1f}s"
                    )
                    if bad_games:
                        print(
                            f"chunk ending at game {count}: {bad_games} game(s) had "
                            f"parse errors and were truncated"
                        )
                    bad_games = 0
                    yield torch.from_numpy(boards), torch.from_numpy(targets)
                    del boards, targets
                    batch = []
                    gc.collect()
                    # resumes here on the next next(), so the training time
                    # between yields is excluded from the next read phase
                    read_start = time.perf_counter()
                    chunk_first_game = count
                    last_report = read_start
                    if count < self.max_games:
                        print(
                            f"  reading games {count + 1:,}-"
                            f"{min(count + self.batchsize, self.max_games):,} "
                            f"(serial, single core)...",
                            flush=True,
                        )

            # the pass is exhausted. read is single-threaded, extract is spread
            # across num_workers, so the serial share bounds what more workers
            # could ever buy
            pass_total = pass_read + pass_extract
            games_read = count - skipped_games
            print(
                f"[pass] done: {games_read:,} games read serially in {pass_read:.1f}s "
                f"({games_read / max(pass_read, 1e-9):,.0f} games/s) | "
                f"extract {pass_extract:.1f}s on {num_workers} workers | "
                f"{pass_positions:,} positions | "
                f"serial share {pass_read / max(pass_total, 1e-9) * 100:.0f}% of {pass_total:.1f}s"
            )


if __name__ == "__main__":
    # test
    # dataset = PGN_Dataset("Team2/pgn_files/Tal.pgn", max_games=10000, batchsize=200)
    # generator = dataset.generate_dataset(6, 256)
    # ratio = 0.9
    # # how many batches makes up the dataset
    # print(dataset.length // dataset.batchsize)
    # # roughly how many batches should be allocated for validation?
    # valid_size = math.ceil(dataset.length // dataset.batchsize * (1 - ratio))
    # print(valid_size)
    # for _ in range(valid_size):
    #     try:
    #         next(generator)
    #     except StopIteration:
    #         raise Exception("what the hell")

    # test yield
    def yield_test():
        yield 1
        yield 2
        yield 3

    generator = yield_test()
    print(next(generator))
    generator = yield_test()
    print(next(generator))

    # start1 = time.process_time()
    # start2 = time.time()
    # while True:
    #     try:
    #         print(len(next(dataset)))
    #     except StopIteration:
    #         break
    # end1 = time.process_time()
    # end2 = time.time()
    # print(f"took {end1-start1} cpu seconds and {end2-start2} clock seconds")

    # test game_to_datapoint
    # with open("Team2/pgn_files/Tal.pgn") as f:
    #     game = chess.pgn.read_game(f)
    #     print(game_to_datapoint(game))
