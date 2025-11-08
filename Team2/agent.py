import torch
import numpy as np
from monte_carlo_tree_search import MonteCarloTreeSearch


class Agent:
    '''
    Agent that uses MCTS with policy and value networks to select moves.
    '''

    def __init__(self, policy_network, value_network, c_puct):
        self.mcts = MonteCarloTreeSearch(policy_network, value_network, c_puct, {}, {}, set())


    def select_move(self, game_state, num_simulations):
        '''
        Selects the best move based on the policy network's predictions.
        '''
        for _ in range(num_simulations): 
            self.mcts.search(game_state)

        return max(self.mcts.frequency_action[game_state], key=self.mcts.frequency_action[game_state].get)

      
    def assign_rewards(self, examples, reward):
        '''
        assigns rewards to each example based on the final game reward
        '''

        # if draw, assign 0 to all
        if reward == 0:
            for example in examples:
                example[2] = 0
        # else, assign alternating rewards (reward for black is opposite of reward for white)
        else:
            examples[-1][2] = reward
            for i in range(2, len(examples)+1):
                examples[-i][2] = -examples[-(i-1)][2] # alternate reward sign
                

    def mcts_self_play(self, num_simulations):
        '''
        executes an iteration of MCTS for the given game state
        num_simulations: number of MCTS simulations to run per move  
        '''
        
        game_state = STARTING_STATE
        examples = [] # stores state, policy, and reward for training
        while True: # infinite loop until terminal state
            # run MCTS simulations to find policy for current state
            for _ in range(num_simulations): 
                self.mcts.search(game_state)
            
            # store training example and find move based on policy
            examples.append([game_state, self.mcts.frequency_action[game_state], None]) # reward to be assigned later
            
            # find optimal move based on visit frequencies (policy)
            freqs = np.array(list(self.mcts.frequency_action[game_state].values()), dtype=np.float32)
            probs = freqs / freqs.sum()
            move = np.random.choice(list(self.mcts.frequency_action[game_state].keys()), p=probs)
            
            game_state = game_state.perform_move(move) # perform selected move
            if game_state.is_terminal(): # check for terminal state
                self.assign_rewards(examples, game_state.get_reward()) # assign rewards to examples
                return examples
            
    
    def training_self_play(self, num_training_iterations, num_games, num_simulations, improvement_threshold):
        '''
        performs self-play training to improve the policy network
        '''
        nn = NEW_POLICY_NETWORK
        examples = []

        # train for specified number of iterations
        for _ in range(num_training_iterations):
            for _ in range(num_games):
                examples += self.mcts_self_play(num_simulations)

            nn = train_policy_network(nn, examples) # train new policy network on examples

            if(pit(nn, self.mcts.policy_network) > improvement_threshold): # if new network is better, update current policy network
                self.mcts.policy_network = nn