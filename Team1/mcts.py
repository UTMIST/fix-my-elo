
class Node():

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS():

    def __init__(self, game, n_simulations):
        self.game = game
        self.n_simulations = n_simulations


    def get_best_move(self, state):
        # Placeholder for selecting the best move after simulations
        pass