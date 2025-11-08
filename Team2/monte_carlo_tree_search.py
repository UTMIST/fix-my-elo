import torch
import math

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
        # base case, terminal state
        if game_state.is_terminal():
            return -game_state.get_reward()
        
        # if state not visited, initialize node with value network
        if game_state not in self.visited:
            self.visited.add(game_state)
            v = self.value_network.predict(game_state)
            return -v


        # select move with highest UCT (upper confidence bound of tree) value
        max_uct, best_move = -float('inf'), None

        # iterate through legal moves to find best UCT
        for move in game_state.get_legal_moves():
            # compute UCT value
            initial_probability = self.policy_network.predict(game_state)[move]
            uct = self.expected_reward[game_state][move] + self.c_puct * initial_probability * math.sqrt(sum(self.frequency_action[game_state])) / (1 + self.frequency_action[game_state][move])
           
            if uct > max_uct:
                max_uct = uct
                best_move = move

        # recursively search the next state
        state_ev = self.search(game_state.perform_move(best_move))

        # update expected reward and frequency action
        self.expected_reward[game_state][best_move] = (self.frequency_action[game_state][best_move] * self.expected_reward[game_state][best_move] + v)/(self.frequency_action[game_state][best_move]+1)
        self.frequency_action[game_state][best_move] += 1

        return -state_ev
    

# some notes:
# - game state needs to have methods is_terminal(), get_reward(), get_legal_moves(), perform_move(move)
# - expected_reward and frequency_action are coded such that game_state is hashable but it has operations so obv this is wrong. needs to be corrected in future