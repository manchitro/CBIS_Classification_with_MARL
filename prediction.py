import torch
import torch.nn as nn

from util import Permute

class Prediction(nn.Module):
    def __init__(self, n_agents: int, n_class: int, hidden_size: int) -> None:
        super().__init__()

        self.n_agents = n_agents
        self.n_class = n_class

        self.linear_sequential = nn.Sequential(
            nn.Linear(self.n_agents, hidden_size),
            nn.GELU(),
			Permute([1, 2, 0]),
			nn.BatchNorm1d(hidden_size),
			Permute([2, 0, 1]),
            nn.Linear(hidden_size, n_class),
        )

        for module in self.linear_sequential:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(cell_state)
