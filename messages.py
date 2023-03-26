import torch
import torch.nn as nn


class MessageSender(nn.Module):
    def __init__(self, belief_size: int, message_size: int,
                 hidden_layer_size_linear: int) -> None:
        super().__init__()
        self.belief_size = belief_size
        self.message_size = message_size
        self.hidden_layer_size_linear = hidden_layer_size_linear

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.belief_size, self.hidden_layer_size_linear),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size_linear, self.message_size)
        )

        for m in self.__seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.__seq_lin(h_t)
