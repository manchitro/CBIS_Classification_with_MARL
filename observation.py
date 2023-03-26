import operator as op
from functools import reduce

import torch


# x = a batch of image, pos = agents' positions in batch, f = agents' window size
def observation(img_batch: torch.Tensor, positions: torch.Tensor, window_size: int) -> torch.Tensor:
    # print("observation witorch image size: ", x.shape, "pos: ", pos.shape, "f: ", f)
    img_sizes = img_batch.size()
    batch_size, side_length = img_sizes[0], img_sizes[1]
    sizes = [size for size in img_sizes[2:]]

    nb_a, _, _ = positions.size()

    pos_min = pos
    pos_max = pos_min + f

    masks = []

    # print("sizes: ", sizes)
    # print("pos_min.shape: ", pos_min.shape)
    # print("pos_max.shape: ", pos_max.shape)
    # print("enumerate(sizes): ", enumerate(sizes))
    for d, s in enumerate(sizes):
        # print("d, s", d, s)
        values = torch.arange(0, s, device=pos.device)

        mask = (pos_min[:, :, d, None] <= values.view(1, 1, s)) & \
               (values.view(1, 1, s) < pos_max[:, :, d, None])

        for n_unsq in range(len(sizes) - 1):
            mask = mask.unsqueeze(-2) if n_unsq < d else mask.unsqueeze(-1)

        masks.append(mask)
    mask = reduce(op.and_, masks)
    mask = mask.unsqueeze(2)

    return x.unsqueeze(0).masked_select(mask) \
        .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
