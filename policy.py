import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, nb_action, n_agents: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1)
        )

        for m in self.__seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_caret_t_next: torch.Tensor) -> torch.Tensor:
        return self.__seq_lin(h_caret_t_next)
