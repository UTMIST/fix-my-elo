import gc
from Team2.data_processing import (
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
        count = 0
        with open(path) as f:
            while count < max_games and chess.pgn.skip_game(f):
                count += 1
        self.length = count

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
            for _ in range(skipped_games):
                # SkipVisitor walks the game without parsing moves, so this
                # costs a file scan rather than a board replay
                if not chess.pgn.skip_game(f):
                    break
            batch = []
            bad_games = 0
            # split the chunk cost into its two phases: pgn parsing in this
            # process, then tensor extraction across the worker pool
            read_start = time.perf_counter()
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
                if count % self.batchsize == 0 or count == self.max_games:
                    read_time = time.perf_counter() - read_start
                    # yields (boards int8 (N,13,8,8), targets int64 (N,2)).
                    extract_start = time.perf_counter()
                    boards, targets = extract_from_fen(
                        batch, num_workers=num_workers, chunksize=chunksize
                    )
                    extract_time = time.perf_counter() - extract_start

                    positions = len(batch)
                    print(
                        f"  chunk to game {count}: {positions:,} positions | "
                        f"pgn read {read_time:6.1f}s | extract {extract_time:6.1f}s "
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
