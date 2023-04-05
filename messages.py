import torch
import torch.nn as nn


class MessageEval(nn.Module):
    def __init__(self, belief_size: int, message_size: int,
                 hidden_layer_size_linear: int) -> None:
        super().__init__()
        self.belief_size = belief_size
        self.message_size = message_size
        self.hidden_layer_size_linear = hidden_layer_size_linear

        self.linear_sequential = nn.Sequential(
            nn.Linear(self.belief_size, self.hidden_layer_size_linear),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size_linear, self.message_size)
        )

        for m in self.linear_sequential:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(h_t)
