"""
HYPERPARAMETERS FOR BOARD ENCODING
"""
import src.export_board

# Game 
BOARD_SIZE = 8
INPUT_SHAPE = (src.export_board.IMAGE_N, BOARD_SIZE, BOARD_SIZE) 
# Total unique moves in chess 
ACTION_SPACE_SIZE = 4672 

# MCTS 
MCTS_SIMULATIONS = 100 # per move
# Exploration constant (C_puct): Higher = explore more, Lower = stick to best moves
C_PUCT = 1.0 

# Training 
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
SELF_PLAY_GAMES = 10