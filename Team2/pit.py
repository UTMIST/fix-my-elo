# run as a module so the Team2 imports resolve: python -m Team2.pit
import torch

from Team2.agent import Agent, pit
from Team2.model_files.SLPolicyValueGPU import SLPolicyValueNetwork

MODEL_PATH_1 = "Team2/model_weights/lab_trained_epoch_3.pth"
MODEL_PATH_2 = "Team2/model_weights/lab_trained_epoch_1.pth"
NUM_GAMES = 10
NUM_SIMULATIONS = 400
# at 0 the search is deterministic and every game comes out identical
TEMPERATURE = 1.0


def load_model(path, device):
    model = SLPolicyValueNetwork().to(device)
    checkpoint = torch.load(path, map_location=device)
    # checkpoints from training are wrapped in a dict, older ones are bare state dicts
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pit] using {device}")

    model1 = load_model(MODEL_PATH_1, device)
    model2 = load_model(MODEL_PATH_2, device)
    print(f"[pit] model1: {MODEL_PATH_1}")
    print(f"[pit] model2: {MODEL_PATH_2}")

    with torch.no_grad():
        # dirichlet_epsilon=0 so the root noise used for self-play exploration
        # does not add randomness to what is meant to be a strength measurement
        score = pit(
            model1,
            model2,
            num_games=NUM_GAMES,
            num_simulations=NUM_SIMULATIONS,
            c_puct=1.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,
            temperature=TEMPERATURE,
            name1=MODEL_PATH_1,
            name2=MODEL_PATH_2,
        )

    print(f"[pit] model1 score: {score:+d} over {NUM_GAMES} games (wins - losses)")
    print("[pit] games written to pit_games/")
