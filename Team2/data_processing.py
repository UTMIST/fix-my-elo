import chess
import chess.pgn
import numpy as np

def encode_fen_to_board(fen):
    """Takes in a FEN string and converts it into an 8x8x12 numpy tensor.
    Uses the key below to map each piece to a 12x1 vector:

    key = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    e.g. if a a white pawn is at e2 (6,4), board[6][4]=[1,0,0,0,0,0,0,0,0,,0,0,0].
    """
    
    key = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    fen = fen.split(" ")[0]
    fen = fen.split("/")

    board = np.zeros((8,8,12),dtype=np.int8)
    
    for rank in range(8):
        col = 0
        while col < len(fen[rank]):
            for char in fen[rank]:
                if char.isdigit():
                    col += int(char)
                else:
                    board[rank][col][key[char]] = 1 # change the value of 12 entry vector
                    col += 1
    return board

def extract_fens_from_pgn(pgn_path):
    """Taken in a path to a pgn file (could contain multiple games) and returns 
    a list containing lists of all positions from each game in FEN format."""
    
    all_fens = []
    while True:
        with open(pgn_path, "r'") as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            fens = []
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())
            all_fens.append(fens)
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
                data_pair = [board.fen(), move]
                game_fens.append(data_pair)
                board.push(move)
                
            all_fens.append(game_fens)
    return all_fens

