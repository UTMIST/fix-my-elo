import math

import chess
import chess.pgn
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
          A tensor of shape (14, 8, 8) representing the board state.
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

class PGNEncoder:
    """Encodes PGN games into positions and moves for training."""

    @staticmethod
    def pgn_to_positions_moves(pgn_file: str, limit: int = 0, min_avg_rating: int = 2000) -> tuple[list[chess.Board], list[chess.Move]]:
        """Convert PGN file to lists of positions and moves for training.
        Args:
            pgn_file (str): Path to the PGN file.
            limit (int): Maximum number of games to process. 0 for no limit.
            min_avg_rating (int): Minimum average rating of players to include the game.
        """
        boards = []
        moves = []
        count = 0

        with open(pgn_file, 'r', encoding='utf-8-sig') as pgn_file:
            while limit == 0 or count <= limit:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                count += 1
                # Extract ratings and result
                white_rating = int(game.headers.get('WhiteRating', 0))
                black_rating = int(game.headers.get('BlackRating', 0))
                avg_rating = (white_rating + black_rating) // 2
                if avg_rating < min_avg_rating:
                    continue

                board = game.board()

                for move in game.mainline_moves():
                    boards.append(board.copy())
                    moves.append(move)
                    board.push(move)

                count += 1
        # print(boards[0:10], moves[0:10])
        return boards, moves

    @staticmethod
    def test_train_split(data: tuple[list[chess.Board], list[chess.Move]], split: float) -> tuple[list[chess.Board], list[chess.Move], list[chess.Board], list[chess.Move]]:
        """Splits the dataset into training and testing sets based on the given split ratio."""
        if not (0 < split < 1):
            raise ValueError("Split ratio must be between 0 and 1.")

        train_boards = []
        train_moves = []
        test_boards = []
        test_moves = []

        boards, moves = data
        split_index = len(boards) * split

        for i in range(len(boards)):
            if i < split_index:
                train_boards.append(boards[i])
                train_moves.append(moves[i])
            else:
                test_boards.append(boards[i])
                test_moves.append(moves[i])

        return train_boards, train_moves, test_boards, test_moves
