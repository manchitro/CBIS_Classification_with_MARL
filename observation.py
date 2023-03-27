import operator as op
from functools import reduce

import torch


def observation(img_batch: torch.Tensor, positions: torch.Tensor, window_size: int) -> torch.Tensor:
    img_sizes = img_batch.size()
    batch_size, side_length = img_sizes[0], img_sizes[1]
    sizes = [size for size in img_sizes[2:]]

    n_agents, _, _ = positions.size()

    pos_min = positions
    pos_max = pos_min + window_size

    masks = []

    for d, s in enumerate(sizes):
        values = torch.arange(0, s, device=positions.device)

        mask = (pos_min[:, :, d, None] <= values.view(1, 1, s)) & \
               (values.view(1, 1, s) < pos_max[:, :, d, None])

        for n_unsq in range(len(sizes) - 1):
            mask = mask.unsqueeze(-2) if n_unsq < d else mask.unsqueeze(-1)

        masks.append(mask)
    mask = reduce(op.and_, masks)
    mask = mask.unsqueeze(2)

    return img_batch.unsqueeze(0).masked_select(mask) \
        .view(n_agents, batch_size, side_length, *[window_size for _ in range(len(sizes))])
