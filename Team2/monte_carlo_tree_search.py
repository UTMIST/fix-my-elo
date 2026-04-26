import math
import torch
import numpy as np
from collections import defaultdict
from data_processing import move_tensor_to_label, uci_to_tensor, fen_to_board_tensor


class Monte_Carlo_Tree_Search:
    '''
    Monte Carlo Tree Search implementation using policy and value networks.
    '''

    def __init__(self, policy_value_network, c_puct, alpha, epsilon, visited, mcts_temperature=1.0, policy_batch_size=16, fpu=2.0):
        '''
        policy_network: neural network that predicts move probabilities
        value_network: neural network that predicts state value
        c_puct: exploration parameter (higher values encourage exploration)
        expected_reward: dictionary mapping game states to expected rewards for each action
        frequency_action: dictionary mapping game states to visit counts for each action
        visited: set of visited game states
        fpu: first play urgency bonus for unvisited moves
        '''
        policy_value_network.eval()
        self.policy_value_network = policy_value_network
        self.device = next(policy_value_network.parameters()).device
        self.c_puct = c_puct
        self.alpha = alpha
        self.epsilon = epsilon
        self.fpu = fpu
        self.mcts_temperature = mcts_temperature
        self.expected_reward = defaultdict(lambda: defaultdict(float)) #all initialize dto 0
        self.frequency_action = defaultdict(lambda: defaultdict(int)) # all initialized to 0
        self.total_visits = defaultdict(int)
        self.visited = visited
        self.policy_cache = {}  
        self.legal_moves_cache = {}
        self.board_tensor_cache = {}
        self.move_label_cache = {}
        self.policy_batch_size = max(1, int(policy_batch_size))
        self.rng = np.random.default_rng()

        self._pending_leaf_boards = []
        self._pending_leaf_paths = []
        self._pending_leaf_tensors = []
        self._pending_leaf_set = set()


    def _get_board_tensor(self, board):
        if board not in self.board_tensor_cache:
            self.board_tensor_cache[board] = fen_to_board_tensor(board)
        return self.board_tensor_cache[board]


    def _get_move_label(self, move):
        label = self.move_label_cache.get(move)
        if label is None:
            label = move_tensor_to_label(uci_to_tensor(move))
            self.move_label_cache[move] = label
        return label


    def _select_best_move(self, board, legal_moves, is_root):
        p = self.policy_cache.get(board)
        if p is None:
            return None

        move_labels = np.fromiter((self._get_move_label(move) for move in legal_moves), dtype=np.int64, count=len(legal_moves)) #mask out illegal moves
        priors = p[0, move_labels].detach().numpy().astype(np.float32, copy=False) / self.mcts_temperature #apply temperature

        if is_root:
            dirichlet_noise = self.rng.dirichlet([self.alpha] * len(legal_moves)).astype(np.float32, copy=False)
            priors = (1 - self.epsilon) * priors + self.epsilon * dirichlet_noise
        
        # priors = np.exp(priors) / np.sum(np.exp(priors))

        freq_board = self.frequency_action[board]
        board_ev = self.expected_reward[board]
        visits = np.fromiter((freq_board[move] for move in legal_moves), dtype=np.float32, count=len(legal_moves))
        q_values = np.fromiter((board_ev[move] for move in legal_moves), dtype=np.float32, count=len(legal_moves))

        # FPU
        q_values += np.where(visits == 0, self.fpu, 0) # if visits=0 add score of 1 to the score

        # vectorized UCT calculation
        sqrt_total_visits = math.sqrt(float(self.total_visits[board]))
        uct_scores = q_values + self.c_puct * priors * sqrt_total_visits / (1.0 + visits)

        return legal_moves[int(np.argmax(uct_scores))]


    def _terminal_value(self, game_state):
        if game_state.result() == "1/2-1/2":
            return 0.0
        return 1.0


    def _select_path_to_leaf(self, game_state):
        path = []
        is_root = True

        while True:
            board = game_state.fen()

            if game_state.is_game_over():
                return "terminal", None, path, self._terminal_value(game_state)

            if board in self._pending_leaf_set:
                return "blocked", None, path, None

            if board not in self.visited:
                board_tensor = self._get_board_tensor(board).unsqueeze(0)
                return "leaf", board, path, board_tensor

            if board in self.legal_moves_cache:
                legal_moves = self.legal_moves_cache[board]
            else:
                legal_moves = [move.uci() for move in game_state.legal_moves]
                self.legal_moves_cache[board] = legal_moves

            best_move = self._select_best_move(board, legal_moves, is_root)

            path.append((board, best_move))
            game_state.push_uci(best_move)
            is_root = False


    def _backup_path(self, path, state_ev):
        for board, move in reversed(path):
            freq = self.frequency_action[board][move]
            ev = self.expected_reward[board][move]
            self.expected_reward[board][move] = (freq * ev + state_ev) / (freq + 1)
            self.frequency_action[board][move] = freq + 1
            self.total_visits[board] += 1
            state_ev = -state_ev


    def _enqueue_leaf(self, board, path, board_tensor):
        self._pending_leaf_boards.append(board)
        self._pending_leaf_paths.append(path)
        self._pending_leaf_tensors.append(board_tensor)
        self._pending_leaf_set.add(board)


    def _flush_pending_leaves(self):
        if not self._pending_leaf_boards:
            return

        batch = torch.cat(self._pending_leaf_tensors, dim=0).to(self.device)
        with torch.no_grad():
            policy_batch, value_batch = self.policy_value_network(batch)

        policy_batch = policy_batch.detach().cpu()
        value_batch = value_batch.detach().cpu()

        for i, board in enumerate(self._pending_leaf_boards):
            self.visited.add(board)
            self.policy_cache[board] = policy_batch[i:i+1]
            leaf_value = float(value_batch[i].item())
            self._backup_path(self._pending_leaf_paths[i], -leaf_value)

        self._pending_leaf_boards.clear()
        self._pending_leaf_paths.clear()
        self._pending_leaf_tensors.clear()
        self._pending_leaf_set.clear()


    def run_simulations(self, game_state, num_simulations, policy_batch_size=None):
        batch_size = self.policy_batch_size if policy_batch_size is None else max(1, int(policy_batch_size))
        simulations_done = 0

        while simulations_done < num_simulations:
            node_type, board, path, payload = self._select_path_to_leaf(game_state.copy())

            if node_type == "terminal":
                self._backup_path(path, float(payload))
                simulations_done += 1
                continue

            if node_type == "leaf":
                self._enqueue_leaf(board, path, payload)
                simulations_done += 1

                if len(self._pending_leaf_boards) >= batch_size:
                    self._flush_pending_leaves()
                continue

            if self._pending_leaf_boards:
                self._flush_pending_leaves()
                continue

            simulations_done += 1

        self._flush_pending_leaves()
    
    
    # OLD MANUAL NON-OPTIMIZED IMPL
    # def search(self, game_state, is_root):
    #     '''
    #     Performs a single iteration (rollout) of MCTS, returning the value of the game state and updating expected_reward and frequency_action.
    #     '''

    #     board = game_state.fen()  # get FEN representation of the board

    #     # base case, terminal state
    #     if game_state.is_game_over():
    #         if game_state.result() == "1/2-1/2":
    #             return 0
    #         # mate returns +1 because it doesn't get negated like other evals when it returns -v or -state_val
    #         return -(-1)
        
    #     # if state not visited, initialize node
    #     if board not in self.visited:
    #         self.visited.add(board)
    #         board_tensor = self._get_board_tensor(board).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             p, v = self.policy_value_network(board_tensor) # calculate value for ev calculation from original state
            
    #         # cache policy for faster runtime
    #         self.policy_cache[board] = p.cpu()
    #         return -v


    #     # select move with highest UCT (upper confidence bound of tree) value
    #     best_move = None


    #     # iterate through legal moves to find best UCT
    #     if board in self.legal_moves_cache:
    #         legal_moves = self.legal_moves_cache[board]
    #     else:
    #         legal_moves = [move.uci() for move in game_state.legal_moves]
    #         self.legal_moves_cache[board] = legal_moves

    #     best_move = self._select_best_move(board, legal_moves, is_root)
    #     if best_move is None:
    #         return 0
            

    #     # recursively search the next state
    #     game_state.push_uci(best_move)
    #     state_ev = self.search(game_state, False)

    #     # update expected reward and frequency action
    #     self.expected_reward[board][best_move] = (self.frequency_action[board][best_move] * self.expected_reward[board][best_move] + state_ev)/(self.frequency_action[board][best_move]+1)
    #     self.frequency_action[board][best_move] += 1
    #     self.total_visits[board] += 1

    #     return -state_ev