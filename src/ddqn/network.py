import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int,
                 layers: Tuple, seed: int = 199):
        """Initialize parameters and build model.
        Params
        ======
            state_size: Dimension of each state
            action_size: Dimension of each action
            layers: Size of each input sample for each layer
            seed: Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0], bias=False)
        self.fc2 = nn.Linear(layers[0], layers[1], bias=False)
        self.fc3 = nn.Linear(layers[1], action_size, bias=False)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x