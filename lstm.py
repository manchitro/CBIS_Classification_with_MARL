from typing import Tuple

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, curr_hidden: torch.Tensor, curr_cell: torch.Tensor, info_unit: torch.Tensor):
        n_agents, batch_size, hidden_size = curr_hidden.size()

        curr_hidden, curr_cell, info_unit = (
            curr_hidden.flatten(0, 1),
            curr_cell.flatten(0, 1),
            info_unit.flatten(0, 1)
        )

		# the nn.LSTM takes the current hidden and cell state and the info_unit
		# and returns the next hidden and cell state
        hidden_next, cell_next = self.lstm(info_unit, (curr_hidden, curr_cell))

        return (
            hidden_next.view(n_agents, batch_size, -1),
            cell_next.view(n_agents, batch_size, -1)
        )
