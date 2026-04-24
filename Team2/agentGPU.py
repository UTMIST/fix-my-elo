# we need to run on a py file instead of a jupyter notebook otherwise multiprocessing will not work properly
from agent import Agent, pit
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import torch
import os

if __name__ == "__main__":
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # MODEL_PATH_1 = os.path.join(SCRIPT_DIR, "model_files", "lab_trained_66.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = SLPolicyValueNetwork().to(device)
    # model1.load_state_dict(torch.load("model_files/SL_stockfish_trained_old.pth", map_location=torch.device("cuda"))["model"])
    model1.load_state_dict(torch.load("SL_trained_stockfish_trained.pth", map_location=torch.device("cuda"))["model"])

    # only train the first model
    agent = Agent(policy_value_network=model1, c_puct=0.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    # agent.training_self_play(num_training_iterations=1, num_games=4, train_to_test_ratio=0.8, num_simulations=10, resign_moves=20, resign_threshold=0.95, num_testing_games=20, improvement_threshold=30, temperature=0.1)

    # example games
    agent.agent_vs_stockfish(num_games=4, num_simulations=1000, path_to_output="pgn_files/examples.pgn", epoch=55)

    # progress too slow? not enough pitting games?
    # examples = agent.stockfish_self_play(2, 1.0)
    # print(examples[0])

    # train fresh network with stockfish
    # agent.stockfish_only_training(iterations=200, num_games=200, train_to_test_ratio=0.8, num_simulations=100, temperature=0.5, workers=20)