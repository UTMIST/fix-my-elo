import chess
import torch


class ChessBoardEncoder:
    """Encodes a chess board into a Tensor for neural network use."""

    # Piece mapping for encoding
    PIECE_TO_INDEX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        """
        Convert chess.Board to a tensor

        Returns:
          A tensor of shape (12, 8, 8) representing the board state.
          - first 6 planes for white pieces
          - next 6 planes for black pieces
          - 1 plane for rep count
          - 1 plane for player to move (1 for white, 0 for black)
        """

        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

        # encode pieces
        for square in chess.SQUARES:
            current_piece = board.piece_at(square)
            if current_piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                piece_index = ChessBoardEncoder.PIECE_TO_INDEX[current_piece.piece_type]

                if current_piece.color == chess.WHITE:
                    tensor[piece_index, rank, file] = 1.0
                else:
                    # since black pieces are stored after white pieces
                    tensor[piece_index + 6, rank, file] = 1.0

                tensor[12, :, :] = 1 if board.is_repetition(2) else 0
                tensor[13, :, :] = 1 if board.turn == chess.WHITE else 0

        return tensor


class MoveEncoder:
    """converts positions, move pair for neural network use and back."""

    def __init__(self):
        self.movetoindex = {}
        self.indextomove = {}
        self._build_mappings()

    def _build_mappings(self):
        """Build mappings from moves to indices and vice versa."""
        index = 0

        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square != to_square:
                    move = chess.Move(from_square, to_square)
                    uci_move = move.uci()
                    self.movetoindex[uci_move] = index
                    self.indextomove[index] = uci_move
                    index += 1

                    possible_promos = [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN]
                    for promotion in possible_promos:
                        move = chess.Move(from_square, to_square, promotion=promotion)
                        uci_move = move.uci()
                        if uci_move not in self.movetoindex:
                            self.movetoindex[uci_move] = index
                            self.indextomove[index] = uci_move
                            index += 1

        self.num_moves = index

    def move_to_index(self, move: chess.Move) -> int:
        """Convert a chess.Move to its corresponding index."""
        uci_move = move.uci()
        return self.movetoindex.get(uci_move, -1)

    def index_to_move(self, index: int) -> str | None:
        """Convert an index back to its corresponding chess.Move."""
        return self.indextomove.get(index, None)