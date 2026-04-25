import torch
import numpy as np
from env import MancalaEnv
from dqn import DQN
from constants import *

env = MancalaEnv()

model = DQN()
model.load_state_dict(torch.load("mancala_model.pth"))
model.eval()


def print_board(board):
    print("\n")
    print("   ", board[5], board[4], board[3], board[2], board[1], board[0])
    print(board[6], "                   ", board[13])
    print("   ", board[7], board[8], board[9], board[10], board[11], board[12])
    print("\n")


def get_ai_action(state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_t).squeeze(0)
    mask = torch.tensor(env.valid_action_mask(), dtype=torch.float32)
    q_values = q_values + (mask - 1) * 1e9
    return int(torch.argmax(q_values).item())


state = env.reset()
done = False

print("You are player 1 (bottom row: pits 0-5)")

while not done:
    print_board(env.board)

    if env.current_player == AGENT:
        # AI turn (top row)
        action_idx = get_ai_action(state)
        print(f"AI chooses: {action_idx}")
        action = action_idx  # AI: 0-5 -> pits 0-5
    else:
        # Human turn (bottom row)
        valid = env.valid_moves()
        print("Valid moves:", valid)  # valid: 0-5 for human
        action_idx = int(input("Choose pit (0-5): "))
        while action_idx not in valid:
            action_idx = int(input("Invalid. Choose again: "))
        action = action_idx + 7  # Human: 0-5 -> pits 7-12

    state, reward, done = env.step(action)

print_board(env.board)

if reward == 1:
    print("AI wins!")
elif reward == -1:
    print("You win!")
else:
    print("Draw!")