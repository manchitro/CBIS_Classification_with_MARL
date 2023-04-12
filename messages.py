import torch
import torch.nn as nn

from util import Permute

class MessageEval(nn.Module):
    def __init__(self, belief_size: int, message_size: int,
                 hidden_layer_size_linear: int) -> None:
        super().__init__()

        self.belief_size = belief_size
        self.message_size = message_size
        self.hidden_layer_size_linear = hidden_layer_size_linear

        self.linear_sequential = nn.Sequential(
            nn.Linear(self.belief_size, self.hidden_layer_size_linear),
            nn.GELU(),
			Permute([1, 2, 0]),
			nn.BatchNorm1d(self.hidden_layer_size_linear),
			Permute([2, 0, 1]),
            nn.Linear(self.hidden_layer_size_linear, self.message_size)
        )

        for module in self.linear_sequential:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(hidden_state)
