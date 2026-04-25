import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 possible actions
        )

    def forward(self, x):
        return self.net(x)