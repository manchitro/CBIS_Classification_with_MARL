import operator as op
import torch

from functools import reduce
from typing import List



def transition(positions: torch.Tensor, next_action_t: torch.Tensor, window_size: int, img_size: List[int]) -> torch.Tensor:
    new_positions = positions.clone()
    dimensions = new_positions.size(-1)

    idxs = []
    for d in range(dimensions):
        idx = (new_positions[:, :, d] + next_action_t[:, :, d] >= 0) * \
              (new_positions[:, :, d] + next_action_t[:, :, d] + window_size < img_size[d])
        idxs.append(idx)

    idx = reduce(op.mul, idxs)

    idx = idx.unsqueeze(2).to(torch.float)

    return idx * (new_positions + next_action_t) + (1 - idx) * new_positions
