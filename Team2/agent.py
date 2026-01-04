import chess
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from monte_carlo_tree_search import Monte_Carlo_Tree_Search
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
from data_processing import fen_to_board_tensor, uci_to_tensor, move_tensor_to_label


# mcts chess agent class
class Agent:
    '''
    Agent that uses MCTS with policy and value networks to select moves.
    '''

    def __init__(self, policy_value_network, c_puct):
        self.policy_value_network = policy_value_network
        self.c_puct = c_puct


    def select_move(self, game_state, num_simulations):
        '''
        Selects the best move based on the policy network's predictions.
        '''
        board = game_state.fen()
        mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, set()) # generate new mcts object to save memory
        for _ in range(num_simulations): 
            mcts.search(game_state.copy())

        return max(mcts.frequency_action[board], key=mcts.frequency_action[board].get)
     

    def mcts_self_play(self, num_simulations):
        '''
        executes an iteration of MCTS for the given game state
        num_simulations: number of MCTS simulations to run per move  
        '''

        game_state = chess.Board()
        examples = [] # stores state, move, and winner for training

        while True: # infinite loop until terminal state
            board = game_state.fen()
            
            # run MCTS simulations to find policy for current state
            mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, set())
            for _ in range(num_simulations): 
                mcts.search(game_state.copy())
            
            # find optimal move based on visit frequencies (policy)
            freqs = np.array(list(mcts.frequency_action[board].values()), dtype=np.float32)
            probs = freqs / freqs.sum()
            move = np.random.default_rng().choice(list(mcts.frequency_action[board].keys()), p=probs)
            
            # store training example
            examples.append([fen_to_board_tensor(board), move_tensor_to_label(uci_to_tensor(move)), None]) # winner to be assigned later
            
            game_state.push_uci(move) # perform selected move
            print(f"move: {len(examples)}, {move} | board: {board}")

            if game_state.is_game_over(): # check for terminal state
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                reward = cases[game_state.result()]
                
                for example in examples: # assign rewards (winners) to examples
                    example[2] = reward 
                    
                return examples
            
    
    def training_self_play(self, num_training_iterations, num_games, train_to_test_ratio, num_simulations, num_testing_games, improvement_threshold):
        '''
        performs self-play training to improve the policy network
        '''
        
        # define policy network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_nn = SLPolicyValueNetwork().to(device)
        new_nn.load_state_dict(torch.load("model_files/sl_policy_value_network4.pth", map_location=torch.device("cpu")))
        policy_criterion = nn.CrossEntropyLoss() # softmax regression loss function
        value_criterion = nn.MSELoss() # use to use logistic loss but expects labels to be 0 or 1, not a range betwen -1 and 1
        optimizer = optim.Adam(new_nn.parameters(), lr=0.1e-4)
        

        # train for specified number of iterations
        examples = []
        
        for epoch in range(num_training_iterations):
            for i in range(num_games): # create training examples through self-play
                print(f"self-play game: {i+1}")
                examples += self.mcts_self_play(num_simulations) # add current game to training examples
                
            # create train and test datasets
            train_dataloader, test_dataloader = examples_to_dataset(examples, train_to_test_ratio)
            
            # train model
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = new_nn(data)  # calculate predictions for this batch
                policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
                value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
                loss = policy_loss + value_loss
                optimizer.zero_grad()  # reset gradient
                loss.backward()  # calculate gradient
                optimizer.step()  # update parameters

                if batch_idx % 100 == 0:
                    print(f"batch progress: epoch {epoch+1} {(100 *(batch_idx +1)/len(train_dataloader)):.2f}% loss: {loss.item():.6f}")
                print(f"batch progress: epoch {epoch+1} {(100 *(batch_idx +1)/len(train_dataloader)):.2f}% loss: {loss.item():.6f}")

            # check validation accuracy to see if general patterns are being learnt
            new_nn.eval()
            test_loss = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_dataloader):
                    data = data.to(device)
                    batch_move_target = target[:, 0].to(device)
                    batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                    pred_policy, pred_val = new_nn(data)
                    policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
                    value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
                    loss = policy_loss + value_loss
                    test_loss += loss

            print('epoch: {}, test loss: {:.6f}'.format(
                epoch + 1,
                test_loss / len(test_dataloader),
                ))
            torch.save({
                "model": new_nn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch+1, 
                "batch": batch_idx,
            }, "checkpoint3.pth")
            print('model checkpoint saved')
    

        # if new network is better, update current policy network
        print('pitting old nn against new nn...')
        pit_result = pit(new_nn, self.policy_value_network, num_testing_games, num_simulations)
        if(pit_result > improvement_threshold): 
            self.policy_value_network = new_nn
            
            print(f'new nn outperformed old nn, {pit_result} games won out of {num_testing_games}')
            print('new nn replaced old nn')
            
        print(f'new nn underperformed old nn, {pit_result} games won out of {num_testing_games}')
        print('new nn did not replace old nn')
            


# agent training helper functions
def pit(policy_value_network1, policy_value_network2, num_games, num_simulations):
    '''
    pit two chess agents w/ two different neural net bases against eachother by playing games, returns how many games nn1 wins against nn2
    '''
    
    agent1 = Agent(policy_value_network=policy_value_network1, c_puct=1.0)
    agent2 = Agent(policy_value_network=policy_value_network2, c_puct=1.0)
    wins = 0
        
    for i in range(num_games): # play num_games games
        print(f'playing testing game {i+1}')
        game_state = chess.Board()
        choice = np.random.default_rng().choice([0, 1]) # random choice for which agent is white or black
        white = [agent1, agent2][choice]
        black = [agent1, agent2][1 - choice] # im sure there's a smarter way to do this lmfao
        
        while True: # infinite loop until terminal state
            # white move
            move = white.select_move(game_state, num_simulations)
            game_state.push_uci(move) # perform selected move
        
            if game_state.is_game_over(): # check for terminal state
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                winner = cases[game_state.result()]
                
                if agent1 == white:
                    wins += max(winner, 0) # 1 -> 1, 0 -> 0, -1 -> 0, again im sure theres a smarter way to do this lol
                else:
                    wins += max(-winner, 0)
                    
                break
            
            
            # black move
            move = black.select_move(game_state, num_simulations)  # run MCTS simulation to find policy for current state
            game_state.push_uci(move) # perform selected move
            
            if game_state.is_game_over(): # check for terminal state
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                winner = cases[game_state.result()]
                
                if agent1 == white:
                    wins += max(winner, 0) # 1 -> 1, 0 -> 0, -1 -> 0, again im sure theres a smarter way to do this lol
                else:
                    wins += max(-winner, 0)
                    
                break

    return wins
    
    
def examples_to_dataset(examples, train_to_test_ratio):
    '''
    format training examples from self-play to training dataset for policy network
    '''
    
    random.shuffle(examples)

    train_size = int(len(examples) * train_to_test_ratio)

    train_data = examples[:train_size] # split the dataset
    test_data = examples[train_size:]

    X_train = torch.stack([board for board, move, winner in train_data])  # (N, 8, 8, 12)
    t_train = torch.tensor([(move, winner) for board, move, winner in train_data])  # (N, 2)

    X_test = torch.stack([board for board, move, winner in test_data])
    t_test = torch.tensor([(move, winner) for board, move, winner in test_data])

    batch_size = 256 # create DataLoaders
    train_dataset = TensorDataset(X_train, t_train)
    test_dataset = TensorDataset(X_test, t_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader