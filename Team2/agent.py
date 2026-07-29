import chess
import chess.pgn
import random
import os
import shutil
from datetime import datetime
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import get_context
import torch.multiprocessing as mp
from Team2.monte_carlo_tree_search import Monte_Carlo_Tree_Search
from Team2.model_files.SLPolicyValueGPU import SLPolicyValueNetwork
from Team2.data_processing import fen_to_board_tensor, uci_to_tensor, move_tensor_to_label
try:
    from stockfish import Stockfish
except ImportError:
    Stockfish = None

# Use file-backed storage for tensor sharing in multiprocessing instead of /dev/shm
mp.set_sharing_strategy('file_system')

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

    def __init__(self, policy_value_network, c_puct, dirichlet_alpha, dirichlet_epsilon, stockfish_path=None):
        self.policy_value_network = policy_value_network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng()
        self.stockfish = None

        preferred_path = stockfish_path or os.getenv("STOCKFISH_PATH")
        default_path = r"stockfish/stockfish-ubuntu-x86-64-avx2"
        resolved_path = preferred_path or (default_path if os.path.exists(default_path) else shutil.which("stockfish"))

        if Stockfish is not None and resolved_path:
            try:
                self.stockfish = Stockfish(path=resolved_path)
            except Exception as e:
                print(f"[Agent] Stockfish init failed at '{resolved_path}': {e}")

    def _require_stockfish(self):
        if self.stockfish is None:
            raise RuntimeError(
                "Stockfish is not initialized. Install python-stockfish and set STOCKFISH_PATH or pass stockfish_path to Agent()."
            )


    def select_move(self, game_state, num_simulations, temperature=0.0, mcts_policy_temperature=1.0, mcts_temperature=1.0, debug=False):
        '''
        Selects the best move based on the policy network's predictions.
        '''
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        board = game_state.fen()
        mcts = Monte_Carlo_Tree_Search(self.policy_value_network, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, set(), mcts_policy_temperature=mcts_policy_temperature, mcts_temperature=mcts_temperature) # generate new mcts object to save memory
        mcts.run_simulations(game_state, num_simulations)

        moves = list(mcts.frequency_action[board].keys())
        counts = np.array(list(mcts.frequency_action[board].values()), dtype=np.float64)

        combined = []
        # debuggning (show move visits + counts)
        if debug:
            if counts.size == 0:
                raise RuntimeError(f"MCTS returned no visit counts for board: {board}")

            # evals/priors must be indexed by `moves`, not by legal_moves: frequency_action
            # is keyed in visit order and only holds moves MCTS actually expanded.
            evals = np.fromiter((mcts.expected_reward[board][move] for move in moves), dtype=np.float32, count=len(moves))
            move_labels = np.fromiter((mcts._get_move_label(move) for move in moves), dtype=np.int64, count=len(moves))
            p = mcts.policy_cache.get(board)
            priors = p[0, move_labels].detach().numpy().astype(np.float32, copy=False)

            combined = zip(moves, counts.tolist(), evals.tolist(), priors.tolist())
            combined = sorted(combined, key=lambda x: x[1], reverse=True)
            checked = sum(1 for item in combined if item[1] > 0)
            # print(f"tested {checked} moves out of ", len(list(game_state.legal_moves)))

        if temperature == 0:
            idx = int(np.argmax(counts))
            return (moves[idx], combined)

        eps = 1e-16
        log_counts = np.log(counts + eps)
        scaled = log_counts / float(temperature)

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

        probs = probs / probs.sum()

        return (self.rng.choice(moves, p=probs), combined)
    
    def evaluate_value(self, fen: str) -> float:
        """Return the scalar value-network evaluation for the given FEN (value is from
        the perspective of the side to move; range roughly -1..+1)."""
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        with torch.no_grad():
            board_tensor = fen_to_board_tensor(fen).unsqueeze(0).to(device)
            _, v = self.policy_value_network(board_tensor)
            return float(v.item())

    def _parse_user_move(self, board: chess.Board, raw_move: str):
        """Try parsing user move as UCI first, then SAN."""
        raw = raw_move.strip()
        if not raw:
            return None

        try:
            candidate = chess.Move.from_uci(raw)
            if candidate in board.legal_moves:
                return candidate
        except ValueError:
            pass

        try:
            return board.parse_san(raw)
        except ValueError:
            return None

    def play_vs_user(
        self,
        num_simulations: int = 200,
        temperature: float = 0.0,
        user_color: str = "white",
        pgn_output_path: str = None,
        starting_fen: str = None,
        debug: bool = False,
    ):
        """
        Interactive terminal game: human vs this agent.

        User commands during their turn:
        - Enter move in UCI (e2e4) or SAN (Nf3, O-O, exd5).
        - help: show commands
        - board: print board again
        - moves: print legal moves (UCI)
        - hint: ask agent for best move for the current side to move
        - resign / quit / exit: resign the game
        """
        if user_color.lower() not in {"white", "black"}:
            raise ValueError("user_color must be 'white' or 'black'")

        board = chess.Board(starting_fen) if starting_fen else chess.Board()
        user_is_white = user_color.lower() == "white"
        user_side = chess.WHITE if user_is_white else chess.BLACK

        game = chess.pgn.Game()
        game.headers["Event"] = "Human vs Team2 Agent"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "User" if user_is_white else "Team2-Agent"
        game.headers["Black"] = "Team2-Agent" if user_is_white else "User"
        if starting_fen:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = starting_fen

        node = game
        resigned = False

        print("Starting Human vs Agent game.")
        print(f"You are playing as {user_color.lower()}.")
        print("Enter 'help' on your turn for commands.\n")

        while not board.is_game_over(claim_draw=True):
            print(board)
            print(f"FEN: {board.fen()}")

            if board.turn == user_side:
                while True:
                    user_input = input("Your move: ").strip()
                    command = user_input.lower()

                    if command == "help":
                        print("Commands: help, board, moves, hint, resign, quit, exit")
                        print("Move formats: e2e4 (UCI) or Nf3 / O-O / exd5 (SAN)")
                        continue

                    if command == "board":
                        print(board)
                        continue

                    if command == "moves":
                        legal = [mv.uci() for mv in board.legal_moves]
                        print("Legal moves:", " ".join(legal))
                        continue

                    if command == "hint":
                        hint = self.select_move(
                            board.copy(stack=False),
                            num_simulations=num_simulations,
                            temperature=0.0,
                            debug=False,
                        )
                        print(f"Agent hint: {str(hint)}")
                        continue

                    if command in {"resign", "quit", "exit"}:
                        resigned = True
                        result = "0-1" if user_is_white else "1-0"
                        game.headers["Result"] = result
                        game.headers["Termination"] = "User resigned"
                        print("You resigned.")
                        break

                    move = self._parse_user_move(board, user_input)
                    if move is None:
                        print("Invalid move. Use UCI like e2e4 or SAN like Nf3.")
                        continue

                    board.push(move)
                    node = node.add_variation(move)
                    print(f"You played: {move.uci()}\n")
                    break

                if resigned:
                    break

            else:
                engine_move = str(
                    self.select_move(
                        board.copy(stack=False),
                        num_simulations=num_simulations,
                        temperature=temperature,
                        debug=debug,
                    )
                )
                board.push_uci(engine_move)
                node = node.add_variation(chess.Move.from_uci(engine_move))
                print(f"Agent played: {engine_move}\n")

        if not resigned:
            game.headers["Result"] = board.result(claim_draw=True)
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                game.headers["Termination"] = outcome.termination.name

        print(board)
        print(f"Game over. Result: {game.headers.get('Result', '*')}")

        if pgn_output_path:
            out_dir = os.path.dirname(pgn_output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(pgn_output_path, "a") as f:
                print(game, file=f, end="\n\n")
            print(f"Saved PGN to {pgn_output_path}")

        return game
    
    def stockfish_only_training(self, iterations, num_games: int, train_to_test_ratio: float, num_simulations: int, temperature: int, workers: int):
        """
        train against stockfish
        """
        self._require_stockfish()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stockfish.set_depth(10)

        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.SmoothL1Loss()
        # value_criterion = nn.MSELoss()
        optimizer = optim.AdamW(
        [{'params': self.policy_value_network.parameters(), 'lr': 1e-3, 'initial_lr': 1e-3}],
        weight_decay=1e-4
        )
        
        scheduler = StepLR(optimizer=optimizer, step_size=50, gamma=0.1, last_epoch=41)

        start_time = time.time()
        recent_epoch_examples = []
        max_epoch_buffer = 50 #30 epochs is around 5gb of data with 100 games per epoch

        for epoch in range(iterations):
            all_examples = []
            print(f"[Stockfish-Only] epoch {epoch}: generating {num_games} games vs Stockfish")

            model_state_dict = {k: v.cpu() for k, v in self.policy_value_network.state_dict().items()}
            worker_args = [
                (model_state_dict, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, num_simulations, temperature)
                for _ in range(num_games)
            ]

            ctx = get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for i, game_examples in enumerate(pool.imap_unordered(stockfish_self_play_worker, worker_args), start=1):
                    all_examples.extend(game_examples)
                    if i % workers == 0 or i == num_games:
                        print(f"  generated {i}/{num_games} games — elapsed: {time.time() - start_time:.2f}s")

            
            recent_epoch_examples.append(all_examples)
            if len(recent_epoch_examples) > max_epoch_buffer:
                recent_epoch_examples.pop(0)

            combined_examples = []
            for ex_list in recent_epoch_examples:
                combined_examples.extend(ex_list)
            train_dataloader, test_dataloader = examples_to_dataset(combined_examples, train_to_test_ratio)

            # Train
            print(f"[Stockfish-Only] training on collected examples using {len(recent_epoch_examples)} epochs of data")
            self.policy_value_network.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)
                pred_policy, pred_val = self.policy_value_network(data)

                mask = (batch_val_target.view(-1) == 1)
                if mask.sum() > 0:
                    policy_loss = policy_criterion(pred_policy[mask], batch_move_target[mask])
                else:
                    policy_loss = torch.tensor(0.0, device=device)

                value_loss = value_criterion(pred_val, batch_val_target)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()

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
            scheduler.step()

            print('epoch: {}, test loss: {:.6f}, lr: {}'.format(
                epoch + 1,
                valid_loss,
                optimizer.param_groups[0]['lr']
                ))
            
            with open("LGB70k_log.txt", "a") as log_file:
                log_file.write(f"elapsed: {time.time() - start_time:.2f}s, epoch: {epoch}, test loss: {valid_loss:.6f}, lr: {optimizer.param_groups[0]['lr']}\n")

            # Checkpoint
            torch.save({
                "model": self.policy_value_network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, "LGB70k_stockfish.pth")
            print("[Stockfish-Only] checkpoint saved: LGB70k_stockfish_trained.pth")

            #Generate examplar game every epoch
            self.agent_vs_stockfish(2, 1000, "pgn_files/LGB70k_examplar_games.pgn", epoch)


    def agent_vs_stockfish(self, num_games, num_simulations, path_to_output, epoch=0, mcts_policy_temperature=1.0, mcts_temperature=1.0):
        """
        export a game between stockfish and the model. stockfish starts first.
        """
        self._require_stockfish()
        self.stockfish.set_depth(15)

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
                    move = self.select_move(game_state=board, num_simulations=num_simulations, temperature=0.0, debug=False,
                                            mcts_policy_temperature=mcts_policy_temperature, mcts_temperature=mcts_temperature)[0]
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
        self._require_stockfish()
        board = chess.Board()
        all = [] # stores state, move, and winner for training
        device = next(self.policy_value_network.parameters()).device
        self.policy_value_network.eval()
        self.stockfish.set_depth(20)
        stockfish_turn = 1
        moves = []

        for i in range(2):
            board = chess.Board()
            examples = []

            if i == 1:
                stockfish_turn = -1
            
            while True:
                # print(board)
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
                    # until move 5, play random moves to create variation
                    if board.fullmove_number < 5:
                        move = random.choice([move.uci() for move in board.legal_moves])
                        board.push_uci(move)
                    else:
                    # start = time.time()
                        self.stockfish.set_fen_position(board.fen())
                        move = self.stockfish.get_best_move()
                        board.push_uci(move)
                    # end = time.time()
                    # print(f"took {end - start:.5f}s for stockfish")
                else:
                    # start = time.time()
                    move = self.select_move(game_state=board, num_simulations=num_simulations, temperature = temperature).item()[0]
                    board.push_uci(move)
                    # end = time.time()
                    # print(f"took {end-start:.5f}s for model")

                examples.append([board_tensor.squeeze(0), move_tensor_to_label(uci_to_tensor(move)), None])
                stockfish_turn *= -1
                moves.append(move)
        
        # Convert tensors to lists before returning to avoid multiprocessing shared storage issues
        for example in all:
            if isinstance(example[0], torch.Tensor):
                example[0] = example[0].tolist()
        
        return all
    

    def mcts_self_play(self, num_simulations, resign_moves, resign_threshold, temperature):
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

            # print(game_state.fullmove_number)
            
            move = self.select_move(game_state, num_simulations=num_simulations, temperature=temperature)[0]
            
            # store training example
            examples.append([board_tensor.squeeze(0), move_tensor_to_label(uci_to_tensor(move)), None]) # winner to be assigned later
            
            game_state.push_uci(move)
            

            # end game ("resign") if value is higher than resign_threshold for resign_moves moves, speeds up training + ensures clean training data for less-trained endgame positions with huge material advantage and not many pieces where moves are pretty random
            # with torch.no_grad():
            #     p, v = self.policy_value_network(board_tensor)
            # v_scalar = v.item()
            # if v_scalar > resign_threshold:
            #     consecutive_high_value_white += 1
            # else:
            #     consecutive_high_value_white = 0
                
            # if v_scalar < -resign_threshold:
            #     consecutive_high_value_black += 1
            # else:
            #     consecutive_high_value_black = 0

            # if consecutive_high_value_white >= resign_moves or consecutive_high_value_black >= resign_moves:
            #     reward = 1 if consecutive_high_value_white >= resign_moves else -1
            #     # assign rewards relative to the player to move at each example
            #     for i, example in enumerate(examples):
            #         multiplier = 1 if (i % 2) == 0 else -1
            #         example[2] = reward * multiplier

            #     return examples
        
        
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
        value_criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(
        [{'params': self.policy_value_network.parameters(), 'lr': 1e-3, 'initial_lr': 1e-3}],
        weight_decay=1e-4
        )

        scheduler = StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
        start_time = time.time()

        recent_epoch_examples = []
        max_epoch_buffer = 3
        workers = 22

        run_name = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(os.path.dirname(__file__), "runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[log] tensorboard writing to {log_dir}")
        # the settings this run's curves came from, so a run is interpretable months
        # later without digging up the commit it was launched from
        writer.add_text(
            "config",
            f"num_training_iterations={num_training_iterations} num_games={num_games} "
            f"num_simulations={num_simulations} temperature={temperature} "
            f"resign_moves={resign_moves} resign_threshold={resign_threshold} "
            f"train_to_test_ratio={train_to_test_ratio} "
            f"lr={optimizer.param_groups[0]['lr']} c_puct={self.c_puct} "
            f"dirichlet_alpha={self.dirichlet_alpha} dirichlet_epsilon={self.dirichlet_epsilon} "
            f"workers={workers} max_epoch_buffer={max_epoch_buffer} "
            f"num_testing_games={num_testing_games} improvement_threshold={improvement_threshold} "
            f"device={device}",
            0,
        )

        checkpoint_path = "self_play_trained.pth"
        games_seen = 0
        positions_seen = 0

        for epoch in range(num_training_iterations):
            all_examples = []
            epoch_start = time.time()
            print(f'starting training iteration {epoch}/{num_training_iterations}')

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
                    temperature
                )
                for _ in range(num_games)
            ]

            ctx = get_context("spawn")  # required for PyTorch safety?

            print(f"[gen] epoch {epoch + 1}: generating {num_games} self-play games on {workers} workers")
            gen_start = time.time()

            with ctx.Pool(processes=workers) as pool:
                for i, game_examples in enumerate(pool.imap_unordered(self_play_worker, args), start=1):
                    all_examples.extend(game_examples)
                    if i % workers == 0 or i == num_games:
                        # rate comes off this epoch's generation only, so it is not
                        # dragged down by time spent training in earlier epochs
                        rate = i / max(time.time() - gen_start, 1e-9)
                        print(
                            f"  [gen] {i}/{num_games} games ({i / num_games * 100:.1f}%) "
                            f"{len(all_examples):,} positions | {rate:.2f} games/s | "
                            f"epoch {time.time() - epoch_start:.1f}s | run {time.time() - start_time:.1f}s"
                        )

            gen_time = time.time() - gen_start
            games_seen += num_games
            positions_seen += len(all_examples)
            print(
                f"[gen] epoch {epoch + 1} done: {num_games} games, {len(all_examples):,} positions "
                f"in {gen_time:.1f}s ({gen_time / max(num_games, 1):.2f}s/game, "
                f"{len(all_examples) / max(num_games, 1):.1f} positions/game)"
            )

            recent_epoch_examples.append(all_examples)
            if len(recent_epoch_examples) > max_epoch_buffer:
                recent_epoch_examples.pop(0)

            combined_examples = []
            for ex_list in recent_epoch_examples:
                combined_examples.extend(ex_list)
            train_dataloader, test_dataloader = examples_to_dataset(combined_examples, train_to_test_ratio)

            num_train_batches = len(train_dataloader)
            print(
                f"[train] epoch {epoch + 1}: {len(combined_examples):,} examples from "
                f"{len(recent_epoch_examples)} epochs of data, {num_train_batches} minibatches"
            )
            self.policy_value_network.train()
            train_policy_sum = 0.0
            train_value_sum = 0.0
            train_loss_sum = 0.0
            train_grad_norm_sum = 0.0
            # positions the policy head actually trains on, since the mask drops
            # every example whose value target is not 1
            masked_positions = 0
            total_positions = 0
            train_batches = 0
            compute_start = time.time()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                batch_move_target = target[:, 0].to(device)
                batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                pred_policy, pred_val = self.policy_value_network(data)

                mask = (batch_val_target.view(-1) == 1)
                if mask.sum() > 0:
                    policy_loss = policy_criterion(pred_policy[mask], batch_move_target[mask])
                else:
                    policy_loss = torch.tensor(0.0, device=device)

                value_loss = value_criterion(pred_val, batch_val_target)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.norm(
                    torch.stack([
                        p.grad.detach().norm(2)
                        for p in self.policy_value_network.parameters()
                        if p.grad is not None
                    ]),
                    2,
                )
                optimizer.step()

                policy_l = policy_loss.item()
                value_l = value_loss.item()
                total_l = policy_l + value_l
                grad_norm_l = grad_norm.item()
                train_policy_sum += policy_l
                train_value_sum += value_l
                train_loss_sum += total_l
                train_grad_norm_sum += grad_norm_l
                masked_positions += int(mask.sum().item())
                total_positions += int(mask.numel())
                train_batches += 1

                if batch_idx % 100 == 0:
                    print(
                        f"  [train] batch {batch_idx + 1}/{num_train_batches} "
                        f"({(batch_idx + 1) / max(num_train_batches, 1) * 100:.1f}%) "
                        f"policy: {policy_l:.4f} value: {value_l:.4f} total: {total_l:.4f} "
                        f"grad_norm: {grad_norm_l:.4f} "
                        f"elapsed: {time.time() - epoch_start:.1f}s"
                    )

            compute_time = time.time() - compute_start

            self.policy_value_network.eval()
            test_policy_sum = 0.0
            test_value_sum = 0.0
            test_loss_sum = 0.0
            eval_start = time.time()

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data = data.to(device)
                    batch_move_target = target[:, 0].to(device)
                    batch_val_target = target[:, 1].float().unsqueeze(1).to(device)

                    pred_policy, pred_val = self.policy_value_network(data)
                    policy_loss = policy_criterion(pred_policy, batch_move_target)  # calculate loss for policy
                    value_loss = value_criterion(pred_val, batch_val_target) # calculate loss for value
                    policy_l = policy_loss.item()
                    value_l = value_loss.item()
                    test_policy_sum += policy_l
                    test_value_sum += value_l
                    test_loss_sum += policy_l + value_l

            eval_time = time.time() - eval_start
            num_test_batches = max(len(test_dataloader), 1)
            valid_policy = test_policy_sum / num_test_batches
            valid_value = test_value_sum / num_test_batches
            valid_loss = test_loss_sum / num_test_batches
            scheduler.step()

            print('epoch: {}, test loss: {:.6f}, lr: {}'.format(
                epoch,
                valid_loss,
                optimizer.param_groups[0]['lr']
                ))
            print(
                "  valid  policy {:.6f} | value {:.6f} | total {:.6f}".format(
                    valid_policy, valid_value, valid_loss
                )
            )
            if train_batches:
                print(
                    "  train  policy {:.6f} | value {:.6f} | total {:.6f} | grad_norm {:.4f}".format(
                        train_policy_sum / train_batches,
                        train_value_sum / train_batches,
                        train_loss_sum / train_batches,
                        train_grad_norm_sum / train_batches,
                    )
                )
            if total_positions:
                print(
                    "  policy head trained on {:,}/{:,} positions ({:.1f}% survived the value==1 mask)".format(
                        masked_positions, total_positions,
                        masked_positions / total_positions * 100,
                    )
                )

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            other = epoch_time - gen_time - compute_time - eval_time
            print(
                "  breakdown: selfplay {:.1f}s ({:.0f}%) | compute {:.1f}s ({:.0f}%) | "
                "eval {:.1f}s ({:.0f}%) | other {:.1f}s | epoch {:.1f}s | run {:.1f}s".format(
                    gen_time, gen_time / max(epoch_time, 1e-9) * 100,
                    compute_time, compute_time / max(epoch_time, 1e-9) * 100,
                    eval_time, eval_time / max(epoch_time, 1e-9) * 100,
                    other, epoch_time, total_time,
                )
            )

            save_start = time.time()
            tmp_path = checkpoint_path + ".tmp"
            torch.save({
                "model": self.policy_value_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch": batch_idx,
                # enough to reconstruct how much self-play these weights saw
                "num_games": num_games,
                "num_simulations": num_simulations,
                "games_seen": games_seen,
                "positions_seen": positions_seen,
            }, tmp_path)
            # rename over the old file so a crash mid-save cannot leave a truncated checkpoint
            os.replace(tmp_path, checkpoint_path)
            print(
                f"[ckpt] epoch {epoch + 1} batch {batch_idx} | {games_seen:,} games / "
                f"{positions_seen:,} positions seen | saved in {time.time() - save_start:.1f}s"
            )

            step = epoch + 1
            if train_batches:
                writer.add_scalar("policy_loss/train", train_policy_sum / train_batches, step)
                writer.add_scalar("value_loss/train", train_value_sum / train_batches, step)
                writer.add_scalar("total_loss/train", train_loss_sum / train_batches, step)
                writer.add_scalar("grad_norm/train", train_grad_norm_sum / train_batches, step)
            writer.add_scalar("policy_loss/valid", valid_policy, step)
            writer.add_scalar("value_loss/valid", valid_value, step)
            writer.add_scalar("total_loss/valid", valid_loss, step)
            writer.add_scalar("time/epoch_seconds", epoch_time, step)
            writer.add_scalar("time/selfplay_seconds", gen_time, step)
            writer.add_scalar("time/compute_seconds", compute_time, step)
            writer.add_scalar("time/eval_seconds", eval_time, step)
            writer.add_scalar("time/total_seconds", total_time, step)
            writer.add_scalar("time/selfplay_fraction", gen_time / max(epoch_time, 1e-9), step)
            writer.add_scalar("data/games_seen", games_seen, step)
            writer.add_scalar("data/positions_seen", positions_seen, step)
            writer.add_scalar("data/positions_this_epoch", len(all_examples), step)
            writer.add_scalar("data/examples_in_buffer", len(combined_examples), step)
            if total_positions:
                writer.add_scalar("data/policy_mask_fraction", masked_positions / total_positions, step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)
            for param_name, param in self.policy_value_network.named_parameters():
                writer.add_histogram(f"weights/{param_name}", param.detach().cpu(), step)
            writer.flush()

            with open("self-play.txt", "a") as log_file:
                log_file.write(f"elapsed: {time.time() - start_time:.2f}s, epoch: {epoch}, test loss: {valid_loss:.6f}, lr: {optimizer.param_groups[0]['lr']}\n")

    
        # if old network is better, update current policy network
        print(f'[pit] pitting old nn against new nn over {num_testing_games} games...')
        pit_start = time.time()
        pit_result = pit(self.policy_value_network, old_nn, num_testing_games, num_simulations, self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon, temperature)
        pit_time = time.time() - pit_start
        print(
            f"[pit] score {pit_result} vs threshold {improvement_threshold} "
            f"in {pit_time:.1f}s | run {time.time() - start_time:.1f}s"
        )
        writer.add_scalar("pit/score", pit_result, num_training_iterations)
        writer.add_scalar("pit/seconds", pit_time, num_training_iterations)
        if(pit_result <= improvement_threshold):
            self.policy_value_network = old_nn
            print(f'new nn underperformed old nn, with score of {pit_result}')
            print('new nn did not replace old nn')

        print(f'new nn outperformed old nn, {pit_result}')
        print('new nn replaced old nn')
        writer.add_text("pit/result", f"score={pit_result} threshold={improvement_threshold}", num_training_iterations)
        writer.flush()
        writer.close()
        print(f"[log] run finished in {time.time() - start_time:.1f}s, tensorboard logs at {log_dir}")



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
        temperature
    ) = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            resign_threshold,
            temperature
        )


def stockfish_self_play_worker(args):
    """Worker for multiprocessing Stockfish-vs-model self-play.

    Expected args: (model_state_dict, c_puct, dirichlet_alpha, dirichlet_epsilon, num_simulations, temperature)
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
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print(f'playing testing game {i}, white: {"agent1" if white == agent1 else "agent2"}, black: {"agent2" if black == agent2 else "agent1"}')

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

        with open(f"pit_games/game_{i:03d}.pgn", "w") as f:
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

    # Convert lists back to tensors if needed
    X_train = torch.stack([torch.tensor(board) if isinstance(board, list) else board for board, move, winner in train_data])  # (N, 13, 8, 8)
    t_train = torch.tensor([(move, winner) for board, move, winner in train_data])  # (N, 2)

    X_test = torch.stack([torch.tensor(board) if isinstance(board, list) else board for board, move, winner in test_data])
    t_test = torch.tensor([(move, winner) for board, move, winner in test_data])

    batch_size = 1024 # create DataLoaders
    train_dataset = TensorDataset(X_train, t_train)
    test_dataset = TensorDataset(X_test, t_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader