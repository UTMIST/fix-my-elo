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
import math
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

        with open(path) as f:
            count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None or count == max_games:
                    break
                count += 1
                if count % 1000 == 0:
                    print(f"counted {count} games")
        self.length = min(count, max_games)

    def generate_dataset(self, num_workers, chunksize):
        # while count <= max_games
        #   load games_per_batch games from path using chess.pgn.readGame
        #   process them all in parallel with helpers to return a batch
        #   yield that batch
        count = 0
        with open(self.path) as f:
            batch = []
            while count <= self.max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                # process this game
                batch += game_to_datapoint(game)

                count += 1
                if count % self.batchsize == 0 or count == self.max_games:
                    processed = extract_from_fen(batch, num_workers=num_workers, chunksize=chunksize)
                    # processed = extract_from_fen_without_multiprocessing(batch)
                    processed = [(torch.from_numpy(board), move, winner) for board, move, winner in processed]
                    yield processed
                    del processed
                    batch = []
                    gc.collect()


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
