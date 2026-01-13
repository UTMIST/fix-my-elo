from minimax import MiniMax
from model_files.SLPolicyValueGPU import SLPolicyValueNetwork
import torch
import chess
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Team2/model_files/LGB_OTB_2025.pth"
model = SLPolicyValueNetwork().to(device)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint["model"])
visited = {}
engine = MiniMax(model, visited)


board = chess.Board("7k/8/R6K/8/2r5/8/8/8 w - - 0 1")

start = time.time()
print(engine.search(board, 3))
end = time.time()
difference = end - start

print(f"time taken: {difference}")
print(f"average time per move calculation: {difference/engine.calculated} seconds")
# print(visited)