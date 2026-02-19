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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
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
        self.stockfish = Stockfish(path=r"stockfish/stockfish-ubuntu-x86-64-avx2")


    def select_move(self, game_state, num_simulations, temperature=0.0): # temperature ONLY for pitting agents against eachother, not for inference (its fine if engine selects dookie move once in a while for pitting because ev is what matters anyways, but in inference we want best move possible)
        '''
        Selects the best move based on the policy network's predictions.
        '''
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        board = game_state.fen()
        mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, set()) # generate new mcts object to save memory
        
        for _ in range(num_simulations): 
            mcts.search(game_state.copy(), True) # perform mcts search

        # apply temperature with numerical stability and NaN-safety
        moves = list(mcts.frequency_action[board].keys())
        counts = np.array(list(mcts.frequency_action[board].values()), dtype=np.float64)

        # combined = zip(moves, counts)
        # combined = sorted(combined, key=lambda x: x[1].item(), reverse=True)

        # # debuggning
        # print(combined)
        # print(self.policy_value_network(fen_to_board_tensor(game_state.fen()).unsqueeze(0).to(device))[1].item())

        if counts.size == 0:
            raise RuntimeError(f"MCTS returned no visit counts for board: {board}")

        # deterministic selection when temperature == 0
        if temperature == 0:
            idx = int(np.argmax(counts))
            return moves[idx]

        # compute a numerically-stable log-softmax over counts
        # add tiny epsilon to avoid log(0)
        eps = 1e-16
        log_counts = np.log(counts + eps)
        scaled = log_counts / float(temperature)

        # fallback to normalized counts if scaling produced non-finite values
        if not np.all(np.isfinite(scaled)):
            probs = counts / counts.sum()
        else:
            # log-sum-exp trick
            m = np.max(scaled)
            exp_scaled = np.exp(scaled - m)
            s = exp_scaled.sum()
            if s <= 0 or not np.isfinite(s):
                probs = counts / counts.sum()
            else:
                probs = exp_scaled / s

        # final sanity: ensure probabilities are finite and sum to 1
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            probs = counts / counts.sum()

        probs = probs / probs.sum()

        return self.rng.choice(moves, p=probs)
    
    def evaluate_value(self, fen: str) -> float:
        """Return the scalar value-network evaluation for the given FEN (value is from
        the perspective of the side to move; range roughly -1..+1)."""
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        with torch.no_grad():
            board_tensor = fen_to_board_tensor(fen).unsqueeze(0).to(device)
            _, v = self.policy_value_network(board_tensor)
            return float(v.item())

    def agent_vs_stockfish(self, num_games, num_simulations, path_to_output, epoch=0):
        """
        export a game between stockfish and the model. stockfish starts first.
        """
        self.stockfish.set_depth(1)

        cpu_start = time.process_time()
        for i in range(num_games):
            board = chess.Board()
            moves = []
            game = chess.pgn.Game()
            game.headers["Event"] = f"Epoch {epoch} Game {i}"
            node = None
            stockfish_turn = (-1)**i

            if stockfish_turn == 1:
                game.headers["White"] = "Stockfish"
                game.headers["Black"] = "Model"
            else:
                game.headers["White"] = "Model"
                game.headers["Black"] = "Stockfish"

            while not board.is_game_over():
                if stockfish_turn == 1:
                    self.stockfish.set_fen_position(board.fen())
                    move = self.stockfish.get_best_move()
                    moves.append(move)
                    board.push_uci(move)
                else:
                    move = self.select_move(game_state=board, num_simulations=num_simulations,temperature=0.1)
                    moves.append(move)
                    board.push_uci(move)

                stockfish_turn *= -1

                if node is None:
                    node = game.add_variation(chess.Move.from_uci(move))
                else:
                    node = node.add_variation(chess.Move.from_uci(move))
            game.headers["Result"] = board.result()
            with open(path_to_output, "a") as file:
                print(game, file=file, end="\n\n")
        
        cpu_end = time.process_time()
        cpu_elapsed = cpu_end - cpu_start
        print(f"took {cpu_elapsed:.4f} seconds")


    def stockfish_self_play(self, num_simulations, temperature):
        """
        Play 1 game where stockfish is white, and another with stockfish as black.
        Returns the games
        """
        board = chess.Board()
        all = [] # stores state, move, and winner for training
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        self.stockfish.set_depth(10)
        stockfish_turn = 1
        moves = []

        for i in range(2):
            board = chess.Board()  # reset board for each game
            examples = []

            if i == 1:
                stockfish_turn = -1
            
            while True:
                if board.is_game_over(): 
                    cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                    reward = cases[board.result()]
                    # assign rewards relative to the player to move at each example
                    if reward == 0:
                        for example in examples:
                            example[2] = 0
                    else:
                        for i, example in enumerate(examples):
                            multiplier = 1 if (i % 2) == 0 else -1
                            example[2] = reward * multiplier

                    all.extend(examples)
                    break
                
                board_fen = board.fen()
                board_tensor = fen_to_board_tensor(board_fen).unsqueeze(0).to(device)

                if stockfish_turn == 1:
                    self.stockfish.set_fen_position(board.fen())
                    move = self.stockfish.get_best_move()
                    board.push_uci(move)
                else:
                    move = self.select_move(game_state=board, num_simulations=num_simulations, temperature = temperature).item()
                    board.push_uci(move)

                examples.append([board_tensor.squeeze(0), move_tensor_to_label(uci_to_tensor(move)), None])
                stockfish_turn *= -1
                moves.append(move)
        
        return all

    def stockfish_only_training(self, iterations, num_games: int, train_to_test_ratio: float, num_simulations: int, temperature: int, workers: int):
        """

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_criterion = nn.CrossEntropyLoss()
        # prefer SmoothL1 (Huber) for value head
        value_criterion = nn.SmoothL1Loss()
        # value_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.policy_value_network.parameters(), lr=1e-4)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        # try cyclic lr for faster learning and possible convergence
        scheduler = CyclicLR(optimizer=optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)

        # backup weights in case training collapses
        backup_state = {k: v.clone().cpu() for k, v in self.policy_value_network.state_dict().items()}

        start_time = time.time()
        # keep a rolling buffer of examples from recent epochs (last N epochs)
        recent_epoch_examples = []
        max_epoch_buffer = 5

        for epoch in range(iterations):
            all_examples = []
            print(f"[Stockfish-Only] epoch {epoch}: generating {num_games} games vs Stockfish")

            # Generate games in parallel
            model_state_dict = {k: v.cpu() for k, v in self.policy_value_network.state_dict().items()}
            worker_args = [
                (model_state_dict, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, num_simulations, temperature)
                for _ in range(num_games)
            ]

            ctx = get_context("spawn")
            output_path = "stockfish_only_examples.pkl"
            with open(output_path, "ab") as f, ctx.Pool(processes=workers) as pool:
                for i, game_examples in enumerate(pool.imap_unordered(stockfish_self_play_worker, worker_args), start=1):
                    all_examples.extend(game_examples)
                    pickle.dump(game_examples, f)
                    if i % workers == 0 or i == num_games:
                        print(f"  generated {i}/{num_games} games — elapsed: {time.time() - start_time:.2f}s")

            

            # add current epoch's examples to recent buffer and concatenate last N epochs
            recent_epoch_examples.append(all_examples)
            if len(recent_epoch_examples) > max_epoch_buffer:
                recent_epoch_examples.pop(0)

            # combine examples from the last up to `max_epoch_buffer` epochs
            combined_examples = []
            for ex_list in recent_epoch_examples:
                combined_examples.extend(ex_list)

            # Build datasets (only drop malformed entries; keep -1 labels so value trains on all examples)
            combined_examples = [ex for ex in combined_examples if len(ex) >= 3]
            train_dataloader, test_dataloader = examples_to_dataset(combined_examples, train_to_test_ratio)

            # Train
            print("[Stockfish-Only] training on collected examples")
            self.policy_value_network.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = self.policy_value_network(data)

                # Policy: only train on datapoints where winner label == 1
                mask = (batch_val_target.view(-1) == 1)
                if mask.sum() > 0:
                    policy_loss = policy_criterion(pred_policy[mask], batch_move_target[mask])
                else:
                    policy_loss = torch.tensor(0.0, device=device)

                # Value: train on all datapoints in the batch
                value_loss = value_criterion(pred_val, batch_val_target)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()

                # gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.policy_value_network.parameters(), max_norm=1.0)

                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"  [train] batch {batch_idx+1}/{len(train_dataloader)} loss: {loss.item():.6f}")

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

            valid_loss = test_loss / len(test_dataloader)
            scheduler.step(valid_loss)

            print('epoch: {}, test loss: {:.6f}, lr: {}'.format(
                epoch + 1,
                valid_loss,
                optimizer.param_groups[0]['lr']
                ))
            
            # test = chess.Board()
            # print(f"eval of initial position: {self.evaluate_value(test.fen())}")

            # Checkpoint
            torch.save({
                "model": self.policy_value_network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, "checkpoint_stockfish_only.pth")
            print("[Stockfish-Only] checkpoint saved: checkpoint_stockfish_only.pth")

            #Generate examplar game every epoch
            self.agent_vs_stockfish(2, 10, "pgn_files/examplar_games.pgn", epoch)

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
                # assign rewards relative to the player to move at each example
                for i, example in enumerate(examples):
                    multiplier = 1 if (i % 2) == 0 else -1
                    example[2] = reward * multiplier

                return examples
        
        
            # end loop with terminal state
            if game_state.is_game_over(): 
                cases = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
                reward = cases[game_state.result()]
                # assign rewards relative to the player to move at each example
                if reward == 0:
                    for example in examples:
                        example[2] = 0
                else:
                    for i, example in enumerate(examples):
                        multiplier = 1 if (i % 2) == 0 else -1
                        example[2] = reward * multiplier

                return examples
            
    
    def training_self_play(self, num_training_iterations, num_games, train_to_test_ratio, num_simulations, resign_moves, resign_threshold, num_testing_games, improvement_threshold, temperature):
        '''
        performs self-play training to improve the policy network
        '''
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        old_nn = SLPolicyValueNetwork().to(device)
        old_nn.load_state_dict(self.policy_value_network.state_dict())
        policy_criterion = nn.CrossEntropyLoss() # softmax regression loss function
        # use SmoothL1 (Huber) for value head - more robust than plain MSE
        value_criterion = nn.SmoothL1Loss()
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

    batch_size = 512 # create DataLoaders
    train_dataset = TensorDataset(X_train, t_train)
    test_dataset = TensorDataset(X_test, t_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader