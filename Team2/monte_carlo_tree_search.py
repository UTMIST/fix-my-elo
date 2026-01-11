import math
import torch
import numpy as np
from collections import defaultdict
from data_processing import move_tensor_to_label, uci_to_tensor, fen_to_board_tensor


class Monte_Carlo_Tree_Search:
    '''
    Monte Carlo Tree Search implementation using policy and value networks.
    '''

    def __init__(self, policy_value_network, c_puct, alpha, epsilon, visited):
        '''
        policy_network: neural network that predicts move probabilities
        value_network: neural network that predicts state value
        c_puct: exploration parameter (higher values encourage exploration)
        expected_reward: dictionary mapping game states to expected rewards for each action
        frequency_action: dictionary mapping game states to visit counts for each action
        visited: set of visited game states
        '''
        policy_value_network.eval()
        self.policy_value_network = policy_value_network
        self.device = next(policy_value_network.parameters()).device
        self.c_puct = c_puct
        self.alpha = alpha
        self.epsilon = epsilon
        self.expected_reward = defaultdict(lambda: defaultdict(float))
        self.frequency_action = defaultdict(lambda: defaultdict(int))
        self.visited = visited
        self.policy_cache = {}  
        self.legal_moves_cache = {}
    
    
    def search(self, game_state, is_root):
        '''
        Performs a single iteration (rollout) of MCTS, returning the value of the game state and updating expected_reward and frequency_action.
        '''

        board = game_state.fen()  # get FEN representation of the board

        # base case, terminal state
        if game_state.is_game_over():
            return -1
        
        # if state not visited, initialize node
        if board not in self.visited:
            self.visited.add(board)
            board_tensor = fen_to_board_tensor(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p, v = self.policy_value_network(board_tensor) # calculate value for ev calculation from original state
            
            # cache policy for faster runtime
            self.policy_cache[board] = p.cpu()
            return -v


        # select move with highest UCT (upper confidence bound of tree) value
        max_uct, best_move = -float('inf'), None


        # iterate through legal moves to find best UCT
        if board in self.legal_moves_cache:
            legal_moves = self.legal_moves_cache[board]
        else:
            legal_moves = [move.uci() for move in game_state.legal_moves]
            self.legal_moves_cache[board] = legal_moves

        # UCT variables
        p = self.policy_cache[board]
        dirichlet_noise = np.random.dirichlet([self.alpha] * len(legal_moves))
        freq_board = self.frequency_action[board]
        board_ev = self.expected_reward[board]
        sqrt_freqs = math.sqrt(sum(freq_board.values()))
        c_puct = self.c_puct
        
        for i, move in enumerate(legal_moves):
            move_label = move_tensor_to_label(uci_to_tensor(move))
            initial_probability = p[0][move_label]
            
            if is_root: # add dirichlet noise for exploration at root node
                initial_probability = (1 - self.epsilon) * initial_probability + self.epsilon * dirichlet_noise[i]
                
            uct = board_ev[move] + c_puct * initial_probability * sqrt_freqs / (1 + freq_board[move]) # compute UCT value
            
            if uct > max_uct: # select move with highest UCT
                max_uct = uct
                best_move = move
            

        # recursively search the next state
        game_state.push_uci(best_move)
        state_ev = self.search(game_state, False)

        # update expected reward and frequency action
        self.expected_reward[board][best_move] = (self.frequency_action[board][best_move] * self.expected_reward[board][best_move] + state_ev)/(self.frequency_action[board][best_move]+1)
        self.frequency_action[board][best_move] += 1

        return -state_ev