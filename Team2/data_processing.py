import chess
import chess.pgn
import torch


def fen_to_board(fen):

    key = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    fen = fen.split(" ")[0]
    fen = fen.split("/")

    board = torch.zeros((8, 8, 12), dtype=torch.int8)

    for rank in range(8):
        col = 0
        while col < len(fen[rank]):
            for char in fen[rank]:
                if char.isdigit():
                    col += int(char)
                else:
                    board[rank][col][
                        key[char]
                    ] = 1  # change the value of 12 entry vector
                    col += 1
    return board


def extract_fens_from_pgn(pgn_path):
    """
    Takes in a path to a pgn file (could contain multiple games) and returns
    a list containing lists of all positions from each game in FEN format."""

    all_fens = []
    with open(pgn_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            game_fens = []
            for move in game.mainline_moves():
                board.push(move)
                game_fens.append(board.fen())
            all_fens.append(game_fens)
    return all_fens


def extract_fens_grouped_with_moves(pgn_path):
    """
    Takes in a pgn path and returns list[list[FEN_in_tensor,move_played]]
    """
    all_fens = []
    with open(pgn_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            game_fens = []
            for move in game.mainline_moves():
                data_pair = [board.fen(), move.uci()]
                game_fens.append(data_pair)
                board.push(move)

            all_fens.append(game_fens)
    return all_fens


def uci_to_tensor(uci_string: str) -> torch.Tensor:
    """Takes in a uci move string and returns a 2x8x8x5 tensor.
    3->board to move from/to + promotion
    8-> rows
    8-> cols
    4-> pawn promotions (none, q,r,b,n)

    E.g. e2e4 -> board[0,4,1,0]=1, board[1,4,3,0]=1 (index 0 is 1 because no promotion happened)
    E.g. e7e8q -> board[0,4,6,0]=1, board[1,4,7,1]=1 (index 1 is 1 because promoted to queen)
    """
    x1, y1 = ord(uci_string[0]) - 97, int(uci_string[1]) - 1
    x2, y2 = ord(uci_string[2]) - 97, int(uci_string[3]) - 1
    promotion = 0

    if len(uci_string) > 4:
        if uci_string[4] == "q":
            promotion = 1
        elif uci_string == "r":
            promotion = 2
        elif uci_string == "b":
            promotion = 3
        else:  # knight promotion
            promotion = 4

    final = torch.zeros(2, 8, 8, 5, dtype=torch.uint8)
    final[0, x1, y1, 0] = 1  # no promotion happens before the move
    final[1, x2, y2, promotion] = 1

    return final


def extract_fens_grouped_with_moves(pgn_path) -> list[str, str]:
    """
    Takes in a pgn path and returns list[list[FEN_in_tensor,move_played]]
    """
    all_fens = []
    with open(pgn_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            game_fens = []
            for move in game.mainline_moves():
                data_pair = [board.fen(), move.uci()]
                game_fens.append(data_pair)
                board.push(move)

            all_fens.append(game_fens)
    return all_fens


def generate_dataset_from_pgn(pgn_path: str) -> list[torch.Tensor, torch.Tensor]:
    """
    Takes in a pgn file path and returns a list of [torch.Tensor(8,8,12), torch.Tensor(2,8,8,4)].
    First tensor is the board state before the move.
    Second tessor is the move made from that position as encoded following uci_to_tensor.
    """
    all_fens = extract_fens_grouped_with_moves(pgn_path)
    dataset = []
    for game in all_fens:
        for data in game:
            fen = data[0]
            move = data[1]

            board = fen_to_board(fen)
            move_tensor = uci_to_tensor(move)

            dataset.append([board, move_tensor])

    return dataset


# test
# dataset = generate_dataset_from_pgn("Team2/masaurus101-white.pgn")
# print(dataset[0])
