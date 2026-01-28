"""
Transforms current board into stack of "images" for model.py's CNN to take

====================================================
LOGIC REFERENCED FROM ALPHAZERO PAPER TY DEEPMIND
====================================================
The final input is a stack of N x 8x8 images.

N = TM + L is composed of...
L:
- 1 layer of only 1s OR 0s denoting which side the player is playing (white == 0)
- 1 layer denoting the total number of turns played in the game so far (from the starting position)
- 1 layer counting number of repeated states in the current game (THREEFOLD REPETITION RULE)
- 1 layer denoting the number of half-moves/no-progress since the last capture or pawn advance (FIFTY-MOVE RULE)
- 2 layers of only 1s OR 0s denoting whether WHITE can castle kingside or queenside
- 2 layers of only 1s OR 0s denoting whether BLACK can castle kingside or queenside
M: always 12
- 6 layers containing only one type piece that is WHITE
- 6 layers containing only one type piece that is BLACK
T: timesteps or the number of board states to include in the prediction (default in AlphaZero is 8)
"""
import chess
import numpy as np

IMAGE_T = 3
IMAGE_L = 6 # accounting for everything included in alpha zero
PLANES_PER_STATE = 2 * 6 # for each state, 6 for WHITE pieces and 6 for BLACK
IMAGE_N = (PLANES_PER_STATE * IMAGE_T) + IMAGE_L

# mappings corresponsing to chess.PieceType and colour
piece_type_to_plane_i = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}
colour_to_i = {
    chess.WHITE: 0,
    chess.BLACK: 1
}

def encode_board(history: list[chess.Board], repeats: int) -> np.array:    
    input = np.zeros((IMAGE_N, 8, 8), np.int8)

    # select slice of history (curr move + previous IMAGE_T-1 moves)
    if len(history) < IMAGE_T:
        state_scope = history
    else:
        state_scope = history[-(IMAGE_T):]

    curr_board = history[-1] # last move recorded

    # loop and fill first IMAGE_T*12 layers (T*M) in "input"
    for t in range(len(state_scope)): # TODO: note the final, current board is added last to input

        start_i = t * PLANES_PER_STATE
        
        for square in range(64):
            if state_scope[t].piece_at(square) is not None: 
                piece = state_scope[t].piece_at(square)

                # relative to the starting index, the first 0-5 planes are WHITE pieces, 6-11 are BLACK pieces
                piece_offset = piece_type_to_plane_i[piece.piece_type]
                color_offset = colour_to_i[piece.color] * 6 # when BLACK, offset by 6
                
                plane_i = piece_offset + color_offset

                row = square // 8
                col = square % 8

                input[start_i + plane_i][row][col] = 1


    # add final L layers

    L_start_i = IMAGE_T * PLANES_PER_STATE

    # 1. player's colour
    # TODO: for now, assume always WHITE
    input[L_start_i][:][:] = 1 # all 1s for WHITE

    #2. total number of turns played so far
    input[L_start_i + 1][:][:] = len(history)

    #3. number of repeated states in current game (threefold repetition rule)
    # TODO: maybe too slow for NN training?
    if curr_board.is_repetition(2): # if current state has occurred twice before
        input[L_start_i + 2][:][:] = 2
    elif curr_board.is_repetition(1): # if current state has occurred once before
        input[L_start_i + 2][:][:] = 1
    elif curr_board.is_repetition(0): # if current state is unique so far
        input[L_start_i + 2][:][:] = 0

    #4. number of half-moves/no-progress since last capture or pawn advance (fifty-move rule)
    input[L_start_i + 3][:][:] = curr_board.halfmove_clock

    #5-6. castling rights for WHITE
    if curr_board.has_kingside_castling_rights(chess.WHITE):
        input[L_start_i + 4][:][:] = 1
    if curr_board.has_queenside_castling_rights(chess.WHITE):
        input[L_start_i + 5][:][:] = 1

    #7-8. castling rights for BLACK
    if curr_board.has_kingside_castling_rights(chess.BLACK):
        input[L_start_i + 6][:][:] = 1
    if curr_board.has_queenside_castling_rights(chess.BLACK):
        input[L_start_i + 7][:][:] = 1


    return input