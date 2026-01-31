import numpy as np
import chess


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a `chess.Board` into a 17×8×8 tensor.

    Tensor encoding plane order:
    0: White pawn
    1: White knight
    2: White bishop
    3: White rook
    4: White queen
    5: White king
    6: Black pawn
    7: Black knight
    8: Black bishop
    9: Black rook
    10: Black queen
    11: Black king
    12: White kingside castling rights
    13: White queenside castling rights
    14: Black kingside castling rights
    15: Black queenside castling rights
    16: En‑passant target file
    """
    # If it is Black’s turn, mirror the board so the network sees the side to move as White
    if board.turn == chess.BLACK:
        mirrored_board = board.mirror()
    else:
        mirrored_board = board.copy()

    planes = 17
    tensor = np.zeros((planes, 8, 8), dtype=np.float32)

    # Map piece types to plane indices for White and Black.
    # Offset Black piece planes by 6 since six planes reserved for White pieces.
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # put pieces into the planes
    for square, piece in mirrored_board.piece_map().items():
        # Compute row and column (Square a1 is 0, h8 is 63)
        row = 7 - (square // 8)
        col = square % 8

        if piece.color == chess.WHITE:
            plane_index = piece_to_plane[piece.piece_type]
        else:
            plane_index = 6 + piece_to_plane[piece.piece_type]

        tensor[plane_index][row][col] = 1.0

    # Castling rights: set all squares in the corresponding plane to 1.0 if that castling right is available
    if mirrored_board.has_kingside_castling_rights(chess.WHITE):
        for r in range(8):
            for c in range(8):
                tensor[12][r][c] = 1.0
    # White queenside
    if mirrored_board.has_queenside_castling_rights(chess.WHITE):
        for r in range(8):
            for c in range(8):
                tensor[13][r][c] = 1.0
    # Black kingside
    if mirrored_board.has_kingside_castling_rights(chess.BLACK):
        for r in range(8):
            for c in range(8):
                tensor[14][r][c] = 1.0
    # Black queenside
    if mirrored_board.has_queenside_castling_rights(chess.BLACK):
        for r in range(8):
            for c in range(8):
                tensor[15][r][c] = 1.0

    # En‑passant: mark the en‑passant square on plane 16
    if mirrored_board.ep_square is not None:
        ep_square = mirrored_board.ep_square
        ep_row = 7 - (ep_square // 8)
        ep_col = ep_square % 8
        tensor[16][ep_row][ep_col] = 1.0

    return tensor


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a chess move into an integer index using AlphaZero encoding.

    This mapping follows the 8×8×73 AlphaZero move representation.  Each
    from‑square (0–63) has 73 possible action planes:

    - 56 sliding planes: 8 directions × 7 distances
    - 8 knight move planes
    - 9 underpromotion planes (3 promotion pieces × 3 directions)

    Underpromotions only include promotions to Knight,
    Bishop, or Rook; promotions to Queen are treated as sliding moves.

    Returns an integer index in 0 <= index < 64*73 representing the move.
    """
 
    # When mirroring a move, from and to squares are mirroed too.
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)
    dr = to_rank - from_rank
    df = to_file - from_file

    # Detect the piece type at the from‑square prior to mirroring
    piece = board.piece_at(move.from_square)

    # For underpromotions, the pawn always moves one rank forward (dr = 1) and optionally one file left or right (df in {‑1, 0, 1}). Map to 9 planes: 3 pieces × 3 directions.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Knight=0, Bishop=1, Rook=2
        prom_piece_map = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
        }
        prom_idx = prom_piece_map.get(move.promotion, 0)
        # Direction index: df = ‑1 → 0, df = 0 → 1, df = +1 → 2
        # Underpromotions always have dr == 1 when viewed from White's perspective
        direction_index = df + 1
        plane_index = 64 + prom_idx * 3 + direction_index
    # Knight moves
    elif piece is not None and piece.piece_type == chess.KNIGHT:
        # Map knight (dr, df) to one of 8 planes.
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        try:
            k_index = knight_moves.index((dr, df))
        except ValueError:
            # If Illegal knight move pattern, fall back to first plane
            k_index = 0
        plane_index = 56 + k_index
    else:
        # Sliding moves Directions are ordered clockwise starting from north: (1,0), (1,1), (0,1), (‑1,1), (‑1,0), (‑1,‑1), (0,‑1), (1,‑1).
        # Distance is 1–7 squares (for king, pawn pushes, queen promotions, distance = 1).
        directions = [
            (1, 0),   # n
            (1, 1),   # ne
            (0, 1),   # e
            (-1, 1),  # se
            (-1, 0),  # s
            (-1, -1), # sw
            (0, -1),  # w
            (1, -1),  # nw
        ]
        direction_index = None
        distance = 0

        for idx, (dr_dir, df_dir) in enumerate(directions):
            # Skip directions that do not point in the same general direction.
            if dr_dir == 0 and df_dir != 0:
                # Horizontal move
                if dr == 0 and df_dir != 0 and df * df_dir > 0:
                    # df_dir indicates sign (+1 east, ‑1 west)
                    distance_candidate = abs(df)
                    direction_index = idx
                    distance = distance_candidate
                    break
            elif df_dir == 0 and dr_dir != 0:
                # Vertical move
                if df == 0 and dr_dir != 0 and dr * dr_dir > 0:
                    distance_candidate = abs(dr)
                    direction_index = idx
                    distance = distance_candidate
                    break
            else:
                # Diagonal moves
                if abs(dr) == abs(df) and dr_dir != 0 and df_dir != 0 and dr * dr_dir > 0 and df * df_dir > 0:
                    distance_candidate = abs(dr)
                    direction_index = idx
                    distance = distance_candidate
                    break

        # ERROR HANDLING:
        # If no direction found, fall back to north with distance 1
        if direction_index is None or distance <= 0:
            direction_index = 0
            distance = 1
        # distance at most 7
        if distance > 7:
            distance = 7
        plane_index = direction_index * 7 + (distance - 1)

    # Combine from‑square and plane index
    final_index = from_sq * 73 + plane_index
    return final_index


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert an integer index back to a chess.Move.

    This function reverses the AlphaZero move encoding used in move_to_index().

    Returns: A chess.Move representing the encoded move.
    """

    from_sq = index // 73
    plane = index % 73

    if board.turn == chess.BLACK:
        actual_from_sq = chess.square_mirror(from_sq)
    else:
        actual_from_sq = from_sq

    # Sliding moves
    if plane < 56:
        direction_index = plane // 7
        distance = (plane % 7) + 1
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        dr_dir, df_dir = directions[direction_index]
        # Compute target square in White orientation
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_rank = from_rank + dr_dir * distance
        to_file = from_file + df_dir * distance

        if to_rank < 0 or to_rank > 7 or to_file < 0 or to_file > 7:
            to_rank = max(0, min(7, to_rank))
            to_file = max(0, min(7, to_file))
        to_sq_white = chess.square(to_file, to_rank)

        # Map back to actual board orientation
        if board.turn == chess.BLACK:
            actual_to_sq = chess.square_mirror(to_sq_white)
        else:
            actual_to_sq = to_sq_white
        return chess.Move(actual_from_sq, actual_to_sq)

    # Knight moves
    if plane < 64:
        k_index = plane - 56
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        dr, df = knight_moves[k_index]
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_rank = from_rank + dr
        to_file = from_file + df
        if to_rank < 0 or to_rank > 7 or to_file < 0 or to_file > 7:
            to_rank = max(0, min(7, to_rank))
            to_file = max(0, min(7, to_file))
        to_sq_white = chess.square(to_file, to_rank)
        if board.turn == chess.BLACK:
            actual_to_sq = chess.square_mirror(to_sq_white)
        else:
            actual_to_sq = to_sq_white
        return chess.Move(actual_from_sq, actual_to_sq)


    # Underpromotion moves
    offset = plane - 64
    prom_piece_index = offset // 3
    direction_index = offset % 3

    # underpromotion pieces: Knight, Bishop, Rook
    prom_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion_piece = prom_pieces[prom_piece_index]

    # Direction
    df = direction_index - 1
    dr = 1
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = from_rank + dr
    to_file = from_file + df

    # bounds
    if to_rank < 0 or to_rank > 7 or to_file < 0 or to_file > 7:
        to_rank = max(0, min(7, to_rank))
        to_file = max(0, min(7, to_file))
    to_sq_white = chess.square(to_file, to_rank)

    # Map back to board
    if board.turn == chess.BLACK:
        actual_to_sq = chess.square_mirror(to_sq_white)
    else:
        actual_to_sq = to_sq_white

    return chess.Move(actual_from_sq, actual_to_sq, promotion=promotion_piece)
