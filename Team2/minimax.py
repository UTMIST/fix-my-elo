import chess
from data_processing import fen_to_board_tensor


class MiniMax:
    """
    MiniMax implementation using only the value head.
    """

    def __init__(self, policy_value_network, visited) -> None:
        """
        policy_network: neural network that predicts move probabilities
        visited: set of visited game states
        """
        self.policy_value_network = policy_value_network
        self.visited = visited
        self.calculated = 0

    def search(self, game_state: chess.Board, depth):
        """
        starting from the root (chess.board), calculates the best move up to depth <depth>
        returns the best move's eval and the move itself.
        """
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        board = game_state.fen()

        self.calculated += 1
        if self.calculated % 100 == 0:
            print(f"checked {self.calculated} moves!")

        # for the player who would've moved, did they win or lose?
        if game_state.outcome() is not None:
            turn = 1
            if not game_state.turn == chess.WHITE:
                turn = -1
            
            if game_state.outcome().winner == chess.WHITE:
                print("win detected!")
                return 1 * turn, None
            elif game_state.outcome().winner == chess.BLACK:
                print("loss detected!")
                return -1 * turn, None
            else:
                print("draw detected!")
                return 0, None

        # if depth limit reached, return eval
        if depth == 0:
            if board in self.visited:
                # print("saved calculation!")
                return self.visited[board]
            board_tensor = fen_to_board_tensor(board).unsqueeze(0).to(device)
            p, v = self.policy_value_network(board_tensor)
            self.visited[board] = (-v.item(), None)
            return -v.item(), None

        # check all moves from current position
        best_eval, best_move = -float("inf"), None
        for move in game_state.legal_moves:
            new_game_state = game_state.copy()
            new_game_state.push(move)
            # flip perspective
            eval = -self.search(new_game_state, depth=depth - 1)[0]

            if best_eval < eval:
                best_eval = eval
                best_move = move

        return best_eval, best_move