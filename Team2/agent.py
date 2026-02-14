import chess
import chess.pgn
import random
import os
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import get_context
from monte_carlo_tree_search import Monte_Carlo_Tree_Search
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
from data_processing import fen_to_board_tensor, uci_to_tensor, move_tensor_to_label
from stockfish import Stockfish

# allow each worker to only use 1 thread to prevent saturation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# mcts chess agent class
class Agent:
    '''
    Agent that uses MCTS with policy and value networks to select moves.
    '''

    def __init__(self, policy_value_network, c_puct, dirichlet_alpha, dirichlet_epsilon):
        self.policy_value_network = policy_value_network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng()


    def select_move(self, game_state, num_simulations, temperature=0.0): # temperature ONLY for pitting agents against eachother, not for inference (its fine if engine selects dookie move once in a while for pitting because ev is what matters anyways, but in inference we want best move possible)
        '''
        Selects the best move based on the policy network's predictions.
        '''
        self.policy_value_network.eval()
        board = game_state.fen()
        mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, set()) # generate new mcts object to save memory
        
        for _ in range(num_simulations): 
            mcts.search(game_state.copy(), True) # perform mcts search

        # apply temperature
        counts = np.array(list(mcts.frequency_action[board].values()), dtype=np.float64)
        log_counts = np.log(counts + 1e-10)  # add epsilon to avoid log(0)
        scaled = log_counts / temperature if temperature > 0 else log_counts * 1e9  # if temperature is 0, make it very large to approximate argmax
        max_scaled = np.max(scaled)  # numerical stability
        exp_scaled = np.exp(scaled - max_scaled)
        probs = exp_scaled / exp_scaled.sum()
        
        return self.rng.choice(list(mcts.frequency_action[board].keys()), p=probs)  # select move based on visit counts 
     

    def stockfish_self_play(self, num_simulations, temperature):
        board = chess.Board()
        examples = [] # stores state, move, and winner for training
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        stockfish = Stockfish(path=r"C:\Users\masar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
        stockfish.set_depth(10)
        stockfish_turn = 1

        moves = []
        while True: # infinite loop until terminal state
            
            board_fen = board.fen()
            board_tensor = fen_to_board_tensor(board_fen).unsqueeze(0).to(device)

            if stockfish_turn == 1:
                stockfish.set_fen_position(board.fen())
                move = stockfish.get_best_move()
                board.push_uci(move)
            else:
                #temperature set to 1.0 (high expoloration)
                move = self.select_move(game_state=board, num_simulations=num_simulations, temperature = temperature).item()
                board.push_uci(move)

            examples.append([board_tensor.squeeze(0), move_tensor_to_label(uci_to_tensor(move)), None])
            stockfish_turn *= -1
            moves.append(move)

            if board.is_game_over(): 
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                reward = cases[board.result()]
                
                for example in examples: # assign rewards (winners) to examples
                    example[2] = reward 
                
                return examples

    def stockfish_only_training(self, iterations, num_games: int, train_to_test_ratio: float, num_simulations: int):
        """

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.policy_value_network.parameters(), lr=1e-5)

        start_time = time.time()

        for epoch in iterations:

            all_examples = []

            print(f"[Stockfish-Only] epoch {epoch}: generating {num_games} games vs Stockfish")

            # Generate games in parallel
            model_state_dict = {k: v.cpu() for k, v in self.policy_value_network.state_dict().items()}
            worker_args = [
                (model_state_dict, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, num_simulations)
                for _ in range(num_games)
            ]

            ctx = get_context("spawn")
            batch_size = 5
            output_path = "stockfish_only_examples.pkl"
            with open(output_path, "ab") as f, ctx.Pool(processes=batch_size) as pool:
                for i, game_examples in enumerate(pool.imap_unordered(stockfish_self_play_worker, worker_args), start=1):
                    all_examples.extend(game_examples)
                    pickle.dump(game_examples, f)
                    if i % batch_size == 0 or i == num_games:
                        print(f"  generated {i}/{num_games} games — elapsed: {time.time() - start_time:.2f}s")

            # Build datasets
            train_dataloader, test_dataloader = examples_to_dataset(all_examples, train_to_test_ratio)

            # Train
            print("[Stockfish-Only] training on collected examples")
            self.policy_value_network.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = self.policy_value_network(data)
                policy_loss = policy_criterion(pred_policy, batch_move_target)
                value_loss = value_criterion(pred_val, batch_val_target)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"  [train] batch {batch_idx+1}/{len(train_dataloader)} loss: {loss.item():.6f}")

            # Checkpoint
            torch.save({
                "model": self.policy_value_network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, "checkpoint_stockfish_only.pth")
            print("[Stockfish-Only] checkpoint saved: checkpoint_stockfish_only.pth")

    def mcts_self_play(self, num_simulations, resign_moves, resign_threshold):
        '''
        executes an iteration of MCTS for the given game state
        num_simulations: number of MCTS simulations to run per move  
        '''

        game_state = chess.Board()
        examples = [] # stores state, move, and winner for training
        consecutive_high_value_white = 0 # counting no. moves with high value position for auto resigning
        consecutive_high_value_black = 0
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        
        while True: # infinite loop until terminal state
            board = game_state.fen()
            board_tensor = fen_to_board_tensor(board).unsqueeze(0).to(device)
            
            mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, set())
            for _ in range(num_simulations): 
                mcts.search(game_state.copy(), True)
            
            freqs = np.array(list(mcts.frequency_action[board].values()), dtype=np.float32)
            probs = freqs / freqs.sum()
            move = self.rng.choice(list(mcts.frequency_action[board].keys()), p=probs)
            
            # store training example
            examples.append([board_tensor.squeeze(0), move_tensor_to_label(uci_to_tensor(move)), None]) # winner to be assigned later
            
            game_state.push_uci(move)
            

            # end game ("resign") if value is higher than resign_threshold for resign_moves moves, speeds up training + ensures clean training data for less-trained endgame positions with huge material advantage and not many pieces where moves are pretty random
            with torch.no_grad():
                p, v = self.policy_value_network(board_tensor)
            v_scalar = v.item()
            if v_scalar > resign_threshold:
                consecutive_high_value_white += 1
            else:
                consecutive_high_value_white = 0
                
            if v_scalar < -resign_threshold:
                consecutive_high_value_black += 1
            else:
                consecutive_high_value_black = 0

            if consecutive_high_value_white >= resign_moves or consecutive_high_value_black >= resign_moves:
                reward = 1 if consecutive_high_value_white >= resign_moves else -1
                for example in examples: # assign rewards (winners) to exmaples
                    example[2] = reward
                    
                return examples
        
        
            # end loop with terminal state
            if game_state.is_game_over(): 
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                reward = cases[game_state.result()]
                
                for example in examples: # assign rewards (winners) to examples
                    example[2] = reward 
                    
                return examples
            
    
    def training_self_play(self, num_training_iterations, num_games, train_to_test_ratio, num_simulations, resign_moves, resign_threshold, num_testing_games, improvement_threshold, temperature):
        '''
        performs self-play training to improve the policy network
        '''
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        old_nn = SLPolicyValueNetwork().to(device)
        old_nn.load_state_dict(self.policy_value_network.state_dict())
        policy_criterion = nn.CrossEntropyLoss() # softmax regression loss function
        value_criterion = nn.MSELoss() # use to use logistic loss but expects labels to be 0 or 1, not a range betwen -1 and 1
        optimizer = optim.Adam(self.policy_value_network.parameters(), lr=0.1e-4)
        start_time = time.time()
        
        examples = []
        
        for epoch in range(num_training_iterations):
            print(f'starting training iteration {epoch+1}/{num_training_iterations}')
            print('--------------------------------')
            
            # create training examples through self-play
            model_state_dict = {
                k: v.cpu()
                for k, v in self.policy_value_network.state_dict().items()
            }

            args = [
                (
                    model_state_dict,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    num_simulations,
                    resign_moves,
                    resign_threshold,
                )
                for _ in range(num_games)
            ]

            ctx = get_context("spawn")  # required for PyTorch safety
            
            batch_size = 4 # fastest on vincent's cpu after a lot of testing
            output_path = "game_examples.pkl" # checkpoint examples
            print(f"completed {0}/{num_games} games, {time.time() - start_time:.2f} seconds elapsed")
            
            with open(output_path, "ab") as f, ctx.Pool(processes=batch_size) as pool:
                for i, game_examples in enumerate(pool.imap_unordered(self_play_worker, args), start=1): # run self-play in parallel
                    examples.extend(game_examples)
                    pickle.dump(game_examples, f)

                    if i % batch_size == 0 or i == num_games:
                        print(f"completed {i}/{num_games} games, {time.time() - start_time:.2f} seconds elapsed")
            

            # create train and test datasets
            train_dataloader, test_dataloader = examples_to_dataset(examples, train_to_test_ratio)
            
            # train model
            print('--------------------------------')
            self.policy_value_network.train()
            
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = self.policy_value_network(data)  # calculate predictions for this batch
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
            self.policy_value_network.eval()
            test_loss = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data = data.to(device)
                    batch_move_target = target[:, 0].to(device)
                    batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                    pred_policy, pred_val = self.policy_value_network(data)
                    policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
                    value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
                    loss = policy_loss + value_loss
                    test_loss += loss

            print('epoch: {}, test loss: {:.6f}'.format(
                epoch + 1,
                test_loss / len(test_dataloader),
                ))
            torch.save({
                "model": self.policy_value_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch+1, 
                "batch": batch_idx,
            }, "checkpoint3.pth")
            print('model checkpoint saved')
            print('--------------------------------')
    

        # if old network is better, update current policy network
        print('pitting old nn against new nn...')
        pit_result = pit(self.policy_value_network, old_nn, num_testing_games, num_simulations, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, temperature)
        if(pit_result <= improvement_threshold): 
            self.policy_value_network = old_nn
            print(f'new nn underperformed old nn, {pit_result} games won out of {num_testing_games}')
            print('new nn did not replace old nn')
            
        print(f'new nn outperformed old nn, {pit_result} games won out of {num_testing_games}')
        print('new nn replaced old nn')



# agent training helper functions
def self_play_worker(args):
    '''
    worker function for multiprocessing self-play
    '''
    (
        model_state_dict,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon,
        num_simulations,
        resign_moves,
        resign_threshold,
    ) = args

    # force CPU
    device = torch.device("cpu")
    # limit threads in worker (safe to call if runtime allows)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    # load model locally
    policy_value_network = SLPolicyValueNetwork().to(device)
    policy_value_network.load_state_dict(model_state_dict)
    policy_value_network.eval()

    agent = Agent(policy_value_network, c_puct, dirichlet_alpha, dirichlet_epsilon)

    with torch.no_grad():
        return agent.mcts_self_play(
            num_simulations,
            resign_moves,
            resign_threshold
        )


def stockfish_self_play_worker(args):
    """Worker for multiprocessing Stockfish-vs-model self-play.

    Expected args: (model_state_dict, c_puct, dirichlet_alpha, dirichlet_epsilon, num_simulations)
    Returns the list of training examples produced by Agent.stockfish_self_play.
    """
    (
        model_state_dict,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon,
        num_simulations,
        temperature,
    ) = args

    # force CPU for worker
    device = torch.device("cpu")
    # limit threads in worker (ignore if not allowed at this point)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    policy_value_network = SLPolicyValueNetwork().to(device)
    policy_value_network.load_state_dict(model_state_dict)
    policy_value_network.eval()

    agent = Agent(policy_value_network, c_puct, dirichlet_alpha, dirichlet_epsilon)

    with torch.no_grad():
        return agent.stockfish_self_play(num_simulations, temperature)


def pit(policy_value_network1, policy_value_network2, num_games, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, temperature):
    '''
    pit two chess agents w/ two different neural net bases against eachother by playing games,
    returns the difference between agent1 wins and agent2 wins
    '''

    os.makedirs("pit_games", exist_ok=True)

    agent1 = Agent(policy_value_network1, c_puct, dirichlet_alpha, dirichlet_epsilon)
    agent2 = Agent(policy_value_network2, c_puct, dirichlet_alpha, dirichlet_epsilon)
    score = 0

    for i in range(num_games):  # play num_games games
        game_state = chess.Board()

        game = chess.pgn.Game()
        node = game

        choice = np.random.default_rng().choice([0, 1])  # random choice for which agent is white or black
        white = [agent1, agent2][choice]
        black = [agent1, agent2][1 - choice]

        game.headers["White"] = "agent1" if white == agent1 else "agent2"
        game.headers["Black"] = "agent2" if black == agent2 else "agent1"

        move_count = 0
        print(f'playing testing game {i+1}, white: {"agent1" if white == agent1 else "agent2"}, black: {"agent2" if black == agent2 else "agent1"}')

        while True:  # infinite loop until terminal state
            # white move
            move = white.select_move(game_state, num_simulations, temperature)
            game_state.push_uci(move)
            node = node.add_variation(chess.Move.from_uci(move))
            move_count += 1

            if game_state.is_game_over():
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                winner = cases[game_state.result()]
                game.headers["Result"] = game_state.result()

                print(f'game {i} over after {move_count} moves, result: {game_state.result()}, outcome: {game_state.outcome().termination.name}, board: {game_state.fen()}')

                if agent1 == white:
                    score += winner
                else:
                    score += -winner

                break

            # black move
            move = black.select_move(game_state, num_simulations, temperature)
            game_state.push_uci(move)
            node = node.add_variation(chess.Move.from_uci(move))
            move_count += 1

            if game_state.is_game_over():
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                winner = cases[game_state.result()]
                game.headers["Result"] = game_state.result()

                print(f'game over after {move_count} moves, result: {game_state.result()}, outcome: {game_state.outcome().termination.name}, board: {game_state.fen()}')

                if agent1 == white:
                    score += winner
                else:
                    score += -winner

                break

        with open(f"pit_games/game_{i+1:03d}.pgn", "w") as f:
            f.write(str(game))

    return score


    
    
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