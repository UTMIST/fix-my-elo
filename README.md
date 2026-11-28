# FixMyElo

FixMyElo, ML Projects 2025-2026.

## Overview
FixMyElo is a policy-value CNN chess engine running MCTS, inspired by AlphaGo/Zero, recreated on a budget. It is strong enough to beat beginner to amateur players on around 800 MCTS sims per move, even [beating Stockfish18 at the lowest depth](model_win_against_stockfish.pgn). We used games from [lumbrasgigabase.com](https://lumbrasgigabase.com/) and [lichess.org](https://lichess.org/) to train our models.

## Architecture
The model architecture closely follows that of Google Deepmind's AlphaZero's. The model takes in as input the encoded board position, then outputs a policy (distribution of possible moves and how viable each is) and a value (static evaluation of the position).
```mermaid
flowchart TD
    Input["Board Input"] --> Conv["Initial Convolution"]
    Conv --> Trunk["10 × Residual Blocks"]

    Trunk --> PolicyHidden["Flatten & Dense Layer"]
    PolicyHidden --> PolicyOut["Policy Output<br/>(Move Scores)"]

    Trunk --> ValueHidden["1×1 Conv & Dense Layer"]
    ValueHidden --> ValueOut["Value Output<br/>(Position Evaluation)"]
```

## MCTS inference
During inference, the model does the following:
1. Create a game tree, with the current position at the root.
2. Calculate the Upper Confidence Bound score for each move.
3. Explore the move with the highest UCB score.
4. Return the move with the most visits made after the simulation budget is used up.

Note that our implementation followed the formula referenced in [this excellent blog](https://suragnair.github.io/posts/alphazero.html)

## Sample position evaluations

| | |
| :---: | :---: |
| <img width="320" alt="Position 1" src="https://github.com/user-attachments/assets/1730d35d-e72b-4074-afc7-94bcea6b25f7" /> | <img width="320" alt="Position 2" src="https://github.com/user-attachments/assets/515311b0-bc3f-4ae3-ac03-7642b6d44978" /> |
| | |
| <img width="320" alt="Position 3" src="https://github.com/user-attachments/assets/1c4f1cc0-6041-4bdd-a06e-429e3345e039" /> | <img width="320" alt="Position 4" src="https://github.com/user-attachments/assets/50b829a9-f6aa-471e-a2a9-116ce84d2ad3" /> |

## TensorBoard logs
The tensorboard log for training can be viewed as follows:
1. Download the log file from [HERE](https://drive.google.com/drive/folders/1VC7vrFbBWn8G_twa3Cyot6SD9RK_KSxU?usp=sharing).
2. Download TensorBoard with `pip install tensorboard`
3. Launch the preview with `tensorboard --logdir 20260728-132218`

<img width="1468" height="790" alt="Screenshot 2026-11-27 at 8 22 05 PM" src="https://github.com/user-attachments/assets/7781458f-57e0-43e8-bd9b-6f473e175bb2" />
<img width="1467" height="789" alt="Screenshot 2026-11-27 at 8 23 18 PM" src="https://github.com/user-attachments/assets/bf4a49cc-bf5e-4fa1-9baa-de6b42e94df2" />
<img width="1470" height="796" alt="Screenshot 2026-11-27 at 8 25 38 PM" src="https://github.com/user-attachments/assets/b0113b97-73d1-4780-9dc2-9ddb25bf9c49" />
<img width="1470" height="805" alt="Screenshot 2026-11-27 at 8 25 15 PM" src="https://github.com/user-attachments/assets/9e049968-edda-4e80-9b61-b128dd0f00eb" />


See the full development notes [HERE](https://docs.google.com/document/d/1r2DsHcfkksEIRd9lvSYco7LtyJas34p0d9OhHjWisBo/edit?tab=t.412b8dl60pdy#heading=h.gel5tpwcy3vn).

## Website Features
1. Play against the engine either as white or black 
3. Set the number of simulations to anything over 2 and less than 2000
4. Visualize the MCTS the engine performed in the tree below the board

Play against it [HERE](https://fme-frontend-795569892145.us-central1.run.app/)!
