import torch
import torch.nn as nn

from util import Permute


class ActionPolicy(nn.Module):
    def __init__(self, n_actions, n_agents: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.linear_sequential = nn.Sequential(
            nn.Linear(n_agents, hidden_size),
            nn.GELU(),
			Permute([1, 2, 0]),
			nn.BatchNorm1d(hidden_size),
			Permute([2, 0, 1]),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        for module in self.linear_sequential:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_state_next_decision: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(hidden_state_next_decision)
