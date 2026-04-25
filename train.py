import torch
import random
from env import MancalaEnv
from dqn import DQN
from replay import ReplayBuffer
from constants import *
import torch.nn as nn
import torch.optim as optim
import numpy as np

env = MancalaEnv()
model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
batch_size = 64

for episode in range(5000):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(env.valid_moves())
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                valid_mask = torch.tensor(env.valid_action_mask(), dtype=torch.float32)
                q_values = q_values + (valid_mask - 1) * 1e9
                action = int(torch.argmax(q_values).item())

        # map action (0-5) to actual pit index
        pit = action + (0 if env.current_player == AGENT else 7)

        next_state, reward, done = env.step(pit)

        buffer.add((state, action, reward, next_state, done))
        state = next_state

        # Train
        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                max_next_q = target_model(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Update target network
    if episode % 50 == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"Episode {episode}, epsilon {epsilon:.3f}")

torch.save(model.state_dict(), "mancala_model.pth")