import torch
import torch.nn as nn


class ActionPolicy(nn.Module):
    def __init__(self, n_actions, n_agents: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.linear_sequential = nn.Sequential(
            nn.Linear(n_agents, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        for m in self.linear_sequential:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_caret_t_next: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(h_caret_t_next)
