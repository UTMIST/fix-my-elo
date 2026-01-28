"""
RUN THE MODEL FROM HERE

Basic game logic to play chess on 8x8 board in the terminal
- player and model class take turns making moves
"""
import chess
from export_board import encode_board

MAX_TURNS = 100

# NN placeholder
class ChessEngine:
    # def __init__(self):
    #     self.move_list = ["g8f6", "e7e5", "d7d5"]  # hardcoded moves
    #     self.move_index = 0

    # def get_move(self, board):
    #     # TODO: change to take in NN output as UCI str
    #     # pick a random legal move
    #     return str(list(board.legal_moves)[0])
    def get_move(self, all_states, turn):
        encoded_board = encode_board(all_states, turn)

# for testing purposes, play the engine against the stockfish engine
# class StockFish:


def main():
    board = chess.Board()
    engine = ChessEngine()
    all_states = []

    print(board)

    max_turns = MAX_TURNS
    turn = 0

    while not board.is_game_over() and turn < max_turns:
        # player moves
        print("\nYour turn! Enter your move in UCI format:")
        player_move = input().strip()
        
        if player_move not in [move.uci() for move in board.legal_moves]:
            print("Invalid move. Try again.")
            continue
        
        board.push_uci(player_move)
        all_states.append(board.copy(stack=False)) # store current state
        
        print(board)

        # checkmate check
        if board.is_game_over(claim_draw=True): #TODO: may get rid of claim_draw if too slow
            print("Game over!")
            break

        # engine moves
        engine_move = engine.get_move(board) # TODO: in future will still take 'board' as an input
        print(f"Engine move: {engine_move}")
        board.push_uci(engine_move)
        all_states.append(board.copy(stack=False)) # store current state
        print(board)

        # checkmate check
        if board.is_game_over(claim_draw=True):
            print("Game over!")
            break

        # update turn count
        turn += 1
        player_move

if __name__ == "__main__":  
    main()
