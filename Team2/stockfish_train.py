from agent import Agent
from stockfish import Stockfish
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import chess
import chess.pgn
import random
import time
import os
import torch

def random_vs_stockfish(num_games):
    stockfish = Stockfish(path=r"C:\Users\masar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
    stockfish.set_depth(10)

    board = chess.Board()
    moves = []

    game = chess.pgn.Game()
    game.headers["Event"] = "Example"
    node = None

    stockfish_turn = 1

    cpu_start = time.process_time()
    for i in range(num_games):
        stockfish_turn = (-1)**i

        while not board.is_game_over():
            if stockfish_turn == 1:
                stockfish.set_fen_position(board.fen())
                move = stockfish.get_best_move()
                moves.append(move)
                board.push_uci(move)
            else:
                legal = list(board.legal_moves)
                move = random.choice(legal).uci()
                moves.append(move)
                board.push_uci(move)

            stockfish_turn *= -1

            if node is None:
                node = game.add_variation(chess.Move.from_uci(move))
            else:
                node = node.add_variation(chess.Move.from_uci(move))
        game.headers["Result"] = board.result()
        print(game)
    
    cpu_end = time.process_time()
    cpu_elapsed = cpu_end - cpu_start
    print(f"took {cpu_elapsed:.4f} seconds")


def agent_vs_stockfish(num_games, num_simulations, path_to_output):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH_1 = os.path.join(SCRIPT_DIR, "model_files", "lab_trained_66.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = SLPolicyValueNetwork().to(device)
    model1.load_state_dict(torch.load(MODEL_PATH_1, map_location=torch.device("cuda"))["model"])
    agent = Agent(policy_value_network=model1, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)

    stockfish = Stockfish(path=r"C:\Users\masar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
    stockfish.set_depth(10)


    cpu_start = time.process_time()
    for i in range(num_games):
        board = chess.Board()
        moves = []
        game = chess.pgn.Game()
        game.headers["Event"] = "Example"
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
                stockfish.set_fen_position(board.fen())
                move = stockfish.get_best_move()
                moves.append(move)
                board.push_uci(move)
            else:
                move = agent.select_move(game_state=board, num_simulations=num_simulations)
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

if __name__ == "__main__":
    agent_vs_stockfish(1000, 2, "pgn_files/stockfish_vs_model.pgn")
    
