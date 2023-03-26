from typing import Tuple

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super.__init__()

        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, hidden: torch.Tensor, cell: torch.Tensor, info_unit: torch.Tensor):
        n_agents, batch_size, hidden_size = hidden.size()

        hidden, cell, info_unit = (
            hidden.flatten(0, 1),
            cell.flatten(0, 1),
            info_unit.flatten(0, 1)
        )

        hidden_next, cell_next = self.lstl(info_unit, (hidden, cell))

        return (
            hidden_next.view(n_agents, batch_size, -1),
            cell_next.view(n_agents, batch_size, -1)
        )
