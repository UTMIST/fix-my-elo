from agent import Agent
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import torch
import os

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model_files", "sl_policy_value_network4.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLPolicyValueNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cuda")))
    agent = Agent(policy_value_network=model, c_puct=0.5)
    agent.training_self_play(num_training_iterations=1, num_games=1000, train_to_test_ratio=0.8, num_simulations=100, resign_moves=20, resign_threshold=0.95, num_testing_games=100, improvement_threshold=30)

    torch.save(agent.policy_value_network.state_dict(), "sl_policy_value_network_from_mcts_self_play.pth")
    # we need to run on a py file instead of a jupyter notebook otherwise multiprocessing will not work properly