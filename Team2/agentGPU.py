# we need to run on a py file instead of a jupyter notebook otherwise multiprocessing will not work properly
from agent import Agent, pit
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import torch
import os

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH_1 = os.path.join(SCRIPT_DIR, "model_files", "sl_policy_value_network5.pth")
    MODEL_PATH_2 = os.path.join(SCRIPT_DIR, "model_files", "sl_policy_value_network_from_mcts_self_play.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = SLPolicyValueNetwork().to(device)
    model1.load_state_dict(torch.load(MODEL_PATH_1, map_location=torch.device("cuda")))
    model2 = SLPolicyValueNetwork().to(device)
    model2.load_state_dict(torch.load(MODEL_PATH_2, map_location=torch.device("cuda")))
    # agent = Agent(policy_value_network=model, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    # agent.training_self_play(num_training_iterations=1, num_games=1000, train_to_test_ratio=0.8, num_simulations=10, resign_moves=20, resign_threshold=0.95, num_testing_games=100, improvement_threshold=30)

    # torch.save(agent.policy_value_network.state_dict(), "sl_policy_value_network_from_mcts_self_play.pth")
    
    
    # test model performance
    pit_result = pit(model2, model1, 10, 100, 1.0, 0.3, 0.25, 0.1)
    print(f'{pit_result} games won out of {10}')