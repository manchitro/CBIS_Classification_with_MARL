import torch
import torch.nn as nn


class Prediction(nn.Module):
    def __init__(self, n_agents: int, n_class: int, hidden_size: int) -> None:
        super().__init__()

        self.n_agents = n_agents
        self.n_class = n_class

        self.linear_sequential = nn.Sequential(
            nn.Linear(self.n_agents, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_class)
        )

        for m in self.linear_sequential:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, c_t: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(c_t)
