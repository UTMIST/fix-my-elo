import chess
import chess.pgn
import torch
import string

# NEED TO ADD COLOUR LAYER IN ENCODING TO INDICATE WHOSE TURN IT IS
def fen_to_board_tensor(fen):

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

    board = torch.zeros((8, 8, 12), dtype=torch.float32)

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
        elif uci_string[4] == "r":
            promotion = 2
        elif uci_string[4] == "b":
            promotion = 3
        else:  # knight promotion
            promotion = 4

    final = torch.zeros(2, 8, 8, 5, dtype=torch.float32)
    final[0, x1, y1, 0] = 1  # no promotion happens before the move
    final[1, x2, y2, promotion] = 1

    return final


def extract_fens_grouped_with_moves(pgn_path, max_games=100) -> list[list[str, str]]:
    """
    Takes in a PGN path and returns list[list[FEN, move_played, winner]]
    Prints progress as "xx% done" without external libraries.
    """
    # First, count total games
    total_games = 0
    with open(pgn_path, "r") as f:
        while chess.pgn.read_game(f):
            total_games += 1
            print(f"Total games: {total_games}")
            if total_games >= max_games:
                break
    total_games = min(total_games, max_games)

    all_fens = []
    with open(pgn_path, "r") as f:
        for i in range(total_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break

            percent = (i + 1) / total_games * 100
            print(f"{percent:.2f}% done")

            board = game.board()
            game_fens = []

            cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
            winner = cases[game.headers["Result"]]

            current_player_eval = 1  # flip evaluation to reflect whether the CURRENT player eventually won or not
            # we want to know if the current player won, as position eval depends on whose turn it is
            for move in game.mainline_moves():
                game_fens.append(
                    [board.fen(), move.uci(), current_player_eval * winner]
                )
                board.push(move)
                current_player_eval *= -1

            all_fens.append(game_fens)

    print("Done!")
    return all_fens


# number of possible encodings with the above is:
# 8x8 (for the first board)
# 8x8x5 (for the second board)
# = 20480 possibilities


def move_tensor_to_label(move_tensor):
    """
    Converts a (2, 8, 8, 5) move tensor to a single integer label (0-20479).

    move_tensor[0, row, col, 0] = 1 indicates from-square
    move_tensor[1, row, col, promo] = 1 indicates to-square and promotion
    """
    # Find the from-square (in the first slice, always at depth 0)
    from_indices = torch.where(move_tensor[0, :, :, 0] == 1)
    from_row = from_indices[0].item()
    from_col = from_indices[1].item()
    from_square = from_row * 8 + from_col  # 0-63

    # Find the to-square and promotion (in the second slice)
    to_indices = torch.where(move_tensor[1] == 1)
    to_row = to_indices[0].item()
    to_col = to_indices[1].item()
    promotion = to_indices[2].item()
    to_square = to_row * 8 + to_col  # 0-63

    # Convert to single label
    label = from_square * 320 + to_square * 5 + promotion

    return label


def label_to_move_table():
    """
    Return a dictionary that maps move label generated by move_tensor_to_label
    back to UCI.
    """
    label_to_move = {}

    for col1 in string.ascii_lowercase[:8]:
        for row1 in range(1, 9):
            for col2 in string.ascii_lowercase[:8]:
                for row2 in range(1, 9):
                    base = f"{col1}{row1}{col2}{row2}"
                    move_label = int(move_tensor_to_label(uci_to_tensor(base)))
                    label_to_move[move_label] = base

                    # if row1 == 7 and row2 == 8:
                    for promotion in ["q", "r", "b", "n"]:
                        new_base = f"{base}{promotion}"
                        move_label = int(move_tensor_to_label(uci_to_tensor(new_base)))
                        label_to_move[move_label] = new_base
    return label_to_move


# print(label_to_move_table()[16554])


def generate_dataset_from_pgn(
    pgn_path: str, max_games=100
) -> list[torch.Tensor, torch.Tensor]:
    """
    Takes in a pgn file path and returns a list of [torch.Tensor(8,8,12), torch.Tensor(2,8,8,5)].
    First tensor is the board state before the move.
    Second tensor is the move made from that position as encoded following uci_to_tensor.
    """
    all_fens = extract_fens_grouped_with_moves(pgn_path, max_games=max_games)
    dataset = []
    for index, game in enumerate(all_fens):
        for data in game:
            fen = data[0]
            move = data[1]
            winner = data[2]

            board = fen_to_board_tensor(fen)
            move_tensor = uci_to_tensor(move)
            move = move_tensor_to_label(move_tensor)

            dataset.append([board, move, winner])

        if index % 100 == 0:
            print(f"Processed {index/len(all_fens)*100}% of games.")

    return dataset


# test
# dataset = generate_dataset_from_pgn("Team2/masaurus101-white.pgn")
# print(dataset[0])
# print(dataset[1])
