import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import chess

from .board import board_to_tensor, move_to_index

ACTION_SIZE = 64 * 73  # 4672


class Node:
    """Node in the MCTS tree."""
    def __init__(self, board: chess.Board, parent: Optional['Node'] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.prior = float(prior)
        self.children: Dict[int, 'Node'] = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

        self.player_turn = board.turn  # True=White, False=Black
        self.value = 0.0  # Value from network at expansion

    def is_terminal(self) -> bool:
        """Terminal if game is over including claimable draws."""
        return self.board.is_game_over(claim_draw=True)

    def expand(self, network: torch.nn.Module):
        """Expand this node using policy/value network with stable legal softmax."""
        # Encode board
        tensor_np = board_to_tensor(self.board)
        tensor = torch.from_numpy(tensor_np).unsqueeze(0).float()

        # Put tensor on same device as network
        net_device = next(network.parameters()).device
        tensor = tensor.to(net_device)

        network.eval()
        with torch.no_grad():
            logits, value = network(tensor)

        self.value = float(value.item())

        logits_np = logits.squeeze(0).detach().to('cpu').numpy()

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            self.is_expanded = True
            return

        # Build (idx, score) for legal moves
        legal_indices: List[int] = []
        legal_scores: List[float] = []
        for mv in legal_moves:
            idx = move_to_index(mv, self.board)
            if 0 <= idx < ACTION_SIZE:
                legal_indices.append(idx)
                s = float(logits_np[idx])

                # handling NaN/Inf
                if not math.isfinite(s):
                    s = 0.0
                legal_scores.append(s)

        if not legal_indices:
            # No encodable legal moves 
            self.is_expanded = True
            return

        # softmax: exp(x - max)
        scores_np = np.asarray(legal_scores, dtype=np.float64)
        m = float(np.max(scores_np))
        exp_scores = np.exp(scores_np - m)
        z = float(np.sum(exp_scores))

        if (not np.isfinite(z)) or (z <= 0.0):
            priors = np.ones(len(legal_indices), dtype=np.float64) / float(len(legal_indices))
        else:
            priors = exp_scores / z

        # Create children
        for mv, idx, p in zip(legal_moves, legal_indices, priors):
            next_board = self.board.copy()
            next_board.push(mv)
            self.children[idx] = Node(next_board, parent=self, prior=float(p))

        self.is_expanded = True

    def compute_ucb(self, child: 'Node', c_puct: float) -> float:
        """Upper Confidence Bound used for selection."""
        q = (child.value_sum / float(child.visit_count)) if child.visit_count > 0 else 0.0
        u = c_puct * child.prior * math.sqrt(self.visit_count + 1e-8) / (1.0 + child.visit_count)
        return q + u

    def select_child(self, c_puct: float) -> Tuple[int, 'Node']:
        """Select child with highest UCB."""
        best_score = -float('inf')
        best_move_idx = None
        best_child = None

        for move_idx, child in self.children.items():
            score = self.compute_ucb(child, c_puct)
            if score > best_score:
                best_score = score
                best_move_idx = move_idx
                best_child = child

        if best_child is None or best_move_idx is None:
            raise RuntimeError("select_child just called on a node with no children.")
        
        return best_move_idx, best_child

    def update(self, value: float, root_player: bool):
        """
        'value' is from root player's perspective (+1 win for root, -1 loss). Flip sign when this node is opponent-to-root.
        """
        self.visit_count += 1
        if self.player_turn == root_player:
            self.value_sum += value
        else:
            self.value_sum -= value


class MCTS:
    """Monte Carlo Tree Search engine."""
    def __init__(self,
                 network: torch.nn.Module,
                 simulations: int = 100,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 exploration_fraction: float = 0.25):
        self.network = network
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.exploration_fraction = float(exploration_fraction)

    def search(self, board: chess.Board) -> List[float]:
        """
        Run MCTS and return a normalized visit-count policy vector.
        """
        root = Node(board.copy(), parent=None, prior=1.0)

        if root.is_terminal():
            return [0.0] * ACTION_SIZE

        root.expand(self.network)

        # Dirichlet noise at root for self-play exploration
        if root.children:
            child_items = list(root.children.items())
            num_children = len(child_items)
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_children).astype(np.float64)

            for i, (_move_idx, child) in enumerate(child_items):
                child.prior = (1.0 - self.exploration_fraction) * child.prior + self.exploration_fraction * float(noise[i])

        root_player = root.player_turn

        for _ in range(self.simulations):
            node = root
            path: List[Node] = [node]

            # Selection
            while node.is_expanded and (not node.is_terminal()):
                if not node.children:
                    break
                _move_idx, node = node.select_child(self.c_puct)
                path.append(node)

            # Expansion
            if (not node.is_terminal()) and (not node.is_expanded):
                node.expand(self.network)

            # Evaluation
            if node.is_terminal():
                result = node.board.result(claim_draw=True)
                if result == '1-0':
                    value = 1.0 if root_player == chess.WHITE else -1.0
                elif result == '0-1':
                    value = 1.0 if root_player == chess.BLACK else -1.0
                else:
                    value = 0.0
            else:
                value = float(node.value)

            # Backprop
            for n in reversed(path):
                n.update(value, root_player)

        # Build policy from visit counts
        policy = [0.0] * ACTION_SIZE
        total_visits = 0

        for move_idx, child in root.children.items():
            v = int(child.visit_count)
            if 0 <= move_idx < ACTION_SIZE:
                policy[move_idx] = float(v)
                total_visits += v

        if total_visits > 0:
            inv = 1.0 / float(total_visits)
            for i in range(ACTION_SIZE):
                policy[i] *= inv

        return policy
