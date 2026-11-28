# FixMyElo

FixMyElo, ML Projects 2025-2026.

## Overview
FixMyElo is a policy-value CNN chess engine running MCTS, inspired by AlphaGo/Zero, recreated on a budget. It is strong enough to beat beginner to amateur players on around 800 MCTS sims per move, even [beating Stockfish18 at the lowest depth](model_win_against_stockfish.pgn).

## Architecture
The model architecture closely follows that of Google Deepmind's AlphaZero's.
```mermaid
flowchart TD
    Input["Board Input"] --> Conv["Initial Convolution"]
    Conv --> Trunk["10 × Residual Blocks"]

    Trunk --> PolicyHidden["Flatten & Dense Layer"]
    PolicyHidden --> PolicyOut["Policy Output<br/>(Move Scores)"]

    Trunk --> ValueHidden["1×1 Conv & Dense Layer"]
    ValueHidden --> ValueOut["Value Output<br/>(Position Evaluation)"]
```

## Sample position evaluations

<img width="424" height="569" alt="Screenshot 2026-11-27 at 7 02 53 PM" src="https://github.com/user-attachments/assets/1730d35d-e72b-4074-afc7-94bcea6b25f7" />
<img width="415" height="551" alt="Screenshot 2026-11-27 at 7 03 17 PM" src="https://github.com/user-attachments/assets/515311b0-bc3f-4ae3-ac03-7642b6d44978" />
<img width="417" height="563" alt="Screenshot 2026-11-27 at 7 03 28 PM" src="https://github.com/user-attachments/assets/1c4f1cc0-6041-4bdd-a06e-429e3345e039" />
<img width="415" height="567" alt="Screenshot 2026-11-27 at 7 07 00 PM" src="https://github.com/user-attachments/assets/50b829a9-f6aa-471e-a2a9-116ce84d2ad3" />


## Features
1. Play against the engine either as white or black 
3. Set the number of simulations to anything over 2 and less than 2000
4. Visualize the MCTS the engine performed in the tree below the board

See the full development notes here: https://docs.google.com/document/d/1r2DsHcfkksEIRd9lvSYco7LtyJas34p0d9OhHjWisBo/edit?tab=t.0#heading=h.zftjrh5ga0lp

Play against it here: https://fme-frontend-795569892145.us-central1.run.app/
