# we need to run on a py file instead of a jupyter notebook otherwise multiprocessing will not work properly
from agent import Agent, pit
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import torch
import os

if __name__ == "__main__":
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # MODEL_PATH_1 = os.path.join(SCRIPT_DIR, "model_files", "lab_trained_66.pth")
    # MODEL_PATH_2 = os.path.join(SCRIPT_DIR, "model_files", "lab_trained_66.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = SLPolicyValueNetwork().to(device)
    model1.load_state_dict(torch.load("checkpoint_stockfish_only.pth", map_location=torch.device("cuda"))["model"])
    # model2 = SLPolicyValueNetwork().to(device)
    # model2.load_state_dict(torch.load(MODEL_PATH_2, map_location=torch.device("cuda"))["model"])

    # only train the first model
    agent = Agent(policy_value_network=model1, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    # agent.training_self_play(num_training_iterations=1, num_games=4, train_to_test_ratio=0.8, num_simulations=10, resign_moves=20, resign_threshold=0.95, num_testing_games=20, improvement_threshold=30, temperature=0.1)

    # progress too slow? not enough pitting games?
    # agent.stockfish_training_self_play(10,30,0.9,10,10,1,10,3,1.0, 10)

    # train fresh network with stockfish
    agent.stockfish_only_training(iterations=50, num_games=30, train_to_test_ratio=0.9, num_simulations=10, temperature=0.5, workers=10)

    # # benchmark
    # import time
    # start = time.process_time()
    # examples = []
    # for _ in range(10):
    #     examples = agent.stockfish_self_play(10)
    # end = time.process_time()
    # print(examples[0], end-start)


    # torch.save(agent.policy_value_network.state_dict(), "lab_self_play.pth")
    
    
    # test model performance
    # pit_result = pit(model2, model1, num_games=10, num_simulations=1, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, temperature=0.1)
    # print(f'{pit_result} games won out of {10}')