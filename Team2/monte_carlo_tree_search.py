import math
from collections import defaultdict
from data_processing import move_tensor_to_label, uci_to_tensor, fen_to_board_tensor


class Monte_Carlo_Tree_Search:
    '''
    Monte Carlo Tree Search implementation using policy and value networks.
    '''

    def __init__(self, policy_value_network, c_puct, visited):
        '''
        policy_network: neural network that predicts move probabilities
        value_network: neural network that predicts state value
        c_puct: exploration parameter (higher values encourage exploration)
        expected_reward: dictionary mapping game states to expected rewards for each action
        frequency_action: dictionary mapping game states to visit counts for each action
        visited: set of visited game states
        '''
        self.policy_value_network = policy_value_network
        self.device = next(policy_value_network.parameters()).device
        self.c_puct = c_puct
        self.expected_reward = defaultdict(lambda: defaultdict(float))
        self.frequency_action = defaultdict(lambda: defaultdict(int))
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
        if board not in self.visited:
            self.visited.add(board)
            board_tensor = fen_to_board_tensor(board).unsqueeze(0).to(self.device)
            p, v = self.policy_value_network(board_tensor)
            return -v


        # select move with highest UCT (upper confidence bound of tree) value
        max_uct, best_move = -float('inf'), None


        # iterate through legal moves to find best UCT
        legal_moves = [move.uci() for move in game_state.legal_moves]
        board_tensor = fen_to_board_tensor(board).unsqueeze(0).to(self.device)
        p, v = self.policy_value_network(board_tensor)

        for move in legal_moves:
            # compute UCT value
            move_label = move_tensor_to_label(uci_to_tensor(move))
            initial_probability = p[0][move_label].item()
            uct = self.expected_reward[board][move] + self.c_puct * initial_probability * math.sqrt(sum(self.frequency_action[board].values())) / (1 + self.frequency_action[board][move])
           
            if uct > max_uct:
                max_uct = uct
                best_move = move


        # recursively search the next state
        game_state.push_uci(best_move)
        state_ev = self.search(game_state)

        # update expected reward and frequency action
        self.expected_reward[board][best_move] = (self.frequency_action[board][best_move] * self.expected_reward[board][best_move] + state_ev)/(self.frequency_action[board][best_move]+1)
        self.frequency_action[board][best_move] += 1

        return -state_ev
    

