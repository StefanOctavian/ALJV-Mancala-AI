import torch
import numpy as np
from env import MancalaEnv
from dqn import DQN
from constants import *
import random

NUM_GAMES = 1000  # Number of games to play

env = MancalaEnv()
model = DQN()
model.load_state_dict(torch.load("mancala_model.pth"))
model.eval()

def get_ai_action(state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_t).squeeze(0)
    mask = torch.tensor(env.valid_action_mask(), dtype=torch.float32)
    q_values = q_values + (mask - 1) * 1e9
    return int(torch.argmax(q_values).item())

ai_wins = 0
random_wins = 0
draws = 0

for game in range(NUM_GAMES):
    state = env.reset()
    done = False

    while not done:
        if env.current_player == AGENT:
            action_idx = get_ai_action(state)
            action = action_idx  # AI: 0-5
        else:
            valid = env.valid_moves()
            action_idx = random.choice(valid)
            action = action_idx + 7  # Human/random: 0-5 -> pits 7-12

        state, reward, done = env.step(action)

    if reward == 1:
        ai_wins += 1
    elif reward == -1:
        random_wins += 1
    else:
        draws += 1

print(f"Out of {NUM_GAMES} games:")
print(f"AI wins: {ai_wins} ({ai_wins/NUM_GAMES:.2%})")
print(f"Random wins: {random_wins} ({random_wins/NUM_GAMES:.2%})")
print(f"Draws: {draws} ({draws/NUM_GAMES:.2%})")