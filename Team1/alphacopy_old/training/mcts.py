"""
Logic for Monte Carlo Tree Search (MCTS) for alphazero clone.

Since the # of actions for a given state is large (tree blows up after a couple moves),
MCTS uses a prob. dist. to only consider the most reasonable moves.

STAGE 1: Selection (function) ==> 
STAGE 2: Expansion ==>
STAGE 3: Simulation (random/heuristic) ==>
"""

import math
import numpy as np
import chess
import config
from src.export_board import encode_board

class Node:
    def __init__(self, prior):
        self.visits = 0
        self.value_sum = 0
        self.prior = prior # P(s,a) from the Neural Net
        self.children = {} # {move: Node}
        self.is_expanded = False

    def value(self):
        # Q(s,a): Average value of this state
        if self.visits == 0: return 0
        return self.value_sum / self.visits

class MCTS:
    def __init__(self, model):
        self.model = model

    def search(self, board):
        root = Node(0)
        
        for _ in range(config.MCTS_SIMULATIONS):
            self._simulate(root, board.copy())
            
        # return visit counts (for policy target)
        return {move: node.visits for move, node in root.children.items()}

    def _simulate(self, node, board):
        # SELECTION ==> traverse down until we hit a leaf
        while node.is_expanded and not board.is_game_over():
            move, node = self._select_child(node)
            board.push(move)

        # EXPANSION & EVALUATION
        if board.is_game_over():
            return self._get_game_result(board)

        # else: expand the node with CNN
        input_tensor = encode_board(board)
        # add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0) 
        
        # get predictions
        policy_preds, value_pred = self.model.predict(input_tensor, verbose=0)
        
        # expand node with valid moves
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            # map move -> index
            move_idx = self._map_move_to_int(move) 
            prior = policy_preds[0][move_idx]
            node.children[move] = Node(prior)
            
        node.is_expanded = True
        
        # BACKPROPAGATION ==> return the value predicted by the NN up the tree
        return value_pred[0][0]

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            # UCB Formula 
            # Q + C * P * (sqrt(Parent_Visits) / (1 + Child_Visits))
            u = config.C_PUCT * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            score = child.value() + u
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
                
        return best_move, best_child
    
    # helpers
    def _get_game_result(self, board):
        """
        Returns game result from the perspective of the current player to move.
        1 = win, -1 = loss, 0 = draw
        """
        result = board.result()
        
        # board.result() returns '1-0' (white wins), '0-1' (black wins), or '1/2-1/2' (draw)
        if result == '1/2-1/2':
            return 0  # draw
        
        # determine who won and who's turn it is
        white_won = result == '1-0'
        white_to_move = board.turn == chess.WHITE
        
        # if it's white's turn and white won, or black's turn and black won
        if (white_to_move and white_won) or (not white_to_move and not white_won):
            return 1  # current player won
        else:
            return -1  

    def _map_move_to_int(self, move):
        """
        Maps a chess.Move to an integer index (0-4671).
        
        Simplified encoding:
        - from_square (0-63): 6 bits
        - to_square (0-63): 6 bits
        - promotion type (0-4): 3 bits for queen, rook, bishop, knight, or none
        
        Total action space: 64 * 64 * 5 = 20,480 (simplified)
        For now, using a simpler mapping: from_square * 64 + to_square
        """
        move_idx = move.from_square * 64 + move.to_square
        
        # handle promotions (add offset for promoted piece type)
        if move.promotion:
            # offset by 4096 (64*64) and add promotion piece type
            # queen=5, rook=4, bishop=3, knight=2
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 4096,
                chess.BISHOP: 8192,
                chess.KNIGHT: 12288
            }
            move_idx += promotion_offset.get(move.promotion, 0)
        
        return move_idx % config.ACTION_SPACE_SIZE  # ensure within bounds