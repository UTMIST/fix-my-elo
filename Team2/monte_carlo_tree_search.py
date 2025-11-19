import math
from data_processing import move_tensor_to_label, uci_to_tensor, fen_to_board


class Monte_Carlo_Tree_Search:
    '''
    Monte Carlo Tree Search implementation using policy and value networks.
    '''

    def __init__(self, policy_network, value_network, c_puct, expected_reward, frequency_action, visited):
        '''
        policy_network: neural network that predicts move probabilities
        value_network: neural network that predicts state value
        c_puct: exploration parameter (higher values encourage exploration)
        expected_reward: dictionary mapping game states to expected rewards for each action
        frequency_action: dictionary mapping game states to visit counts for each action
        visited: set of visited game states
        '''
        self.policy_network = policy_network
        self.value_network = value_network
        self.c_puct = c_puct
        self.expected_reward = expected_reward
        self.frequency_action = frequency_action
        self.visited = visited
    
    
    def search(self, game_state):
        '''
        Performs a single iteration of MCTS, returning the value of the game state and updating expected_reward and frequency_action.
        '''

        board = game_state.fen()  # get FEN representation of the board

        # base case, terminal state
        if game_state.is_game_over():
            return -1
        
        # if state not visited, initialize node with value network
        if game_state not in self.visited:
            self.visited.add(board)
            v = self.value_network.predict(fen_to_board(board))
            return -v


        # select move with highest UCT (upper confidence bound of tree) value
        max_uct, best_move = -float('inf'), None


        # iterate through legal moves to find best UCT
        legal_moves = [move.uci() for move in game_state.legal_moves]

        for move in legal_moves:
            # compute UCT value
            move_label = move_tensor_to_label(uci_to_tensor(move))
            initial_probability = self.policy_network.predict(fen_to_board(board))[move_label]
            uct = self.expected_reward[board][move] + self.c_puct * initial_probability * math.sqrt(sum(self.frequency_action[board])) / (1 + self.frequency_action[board][move])
           
            if uct > max_uct:
                max_uct = uct
                best_move = move


        # recursively search the next state
        state_ev = self.search(game_state.push_uci(best_move))

        # update expected reward and frequency action
        self.expected_reward[board][best_move] = (self.frequency_action[board][best_move] * self.expected_reward[board][best_move] + v)/(self.frequency_action[board][best_move]+1)
        self.frequency_action[board][best_move] += 1

        return -state_ev
    

