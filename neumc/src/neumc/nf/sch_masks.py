from collections.abc import Sequence
import numpy as np
import torch
from typing import Generator
from .u1_masks import (
    make_single_stripes,
    make_double_stripes,
    make_2d_link_active_stripes,
    make_plaq_masks,
)


def make_mask(
    shape: Sequence[int],
    mu: int,
    period: int,
    row_offsets: Sequence[int],
    offset: int,
    float_dtype,
    device,
) -> torch.Tensor:
    nu = 1 - mu
    if mu == 0:
        n_rows = shape[1]
        n_cols = shape[0]
    else:
        n_rows = shape[0]
        n_cols = shape[1]

    row = np.zeros(n_cols)
    row[::period] = 1

    rows = []
    r_period = len(row_offsets)

    for i in range(n_rows):
        rows.append(np.roll(row, row_offsets[i % r_period]))

    mask = np.stack(rows, nu)

    mask = np.roll(mask, offset, mu)
    return torch.from_numpy(mask.astype(float_dtype)).to(device)


def make_schwinger_plaq_mask(
    shape: Sequence[int], mu: int, offset: int, float_dtype, device
) -> dict[str, torch.Tensor]:
    mask = {}
    mask["frozen"] = make_mask(
        shape, mu, 2, [0], offset + 1, float_dtype=float_dtype, device=device
    )
    mask["active"] = make_mask(
        shape, mu, 4, [0, 2], offset, float_dtype=float_dtype, device=device
    )
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def make_schwinger_link_mask(
    shape: Sequence[int], mu: int, offset: int, float_dtype, device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device)
    mask[mu] = make_mask(shape[1:], mu, 4, [0, 2], offset, float_dtype, device)

    return mask


def schwinger_masks(
    *,
    plaq_mask_shape: Sequence[int],
    link_mask_shape: Sequence[int],
    float_dtype,
    device,
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor]]], None, None]:
    i = 0
    while True:
        # periodically loop through all arrangements of maskings
        mu = (i // 4) % 2
        off = i % 4

        link_mask = make_schwinger_link_mask(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_schwinger_plaq_mask(
            plaq_mask_shape, mu, off, float_dtype=float_dtype, device=device
        )

        yield link_mask, (plaq_mask,)
        i += 1


def schwinger_masks_with_2x1_loops(
    *,
    plaq_mask_shape: Sequence[int],
    link_mask_shape: Sequence[int],
    float_dtype,
    device,
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]], None, None]:
    i = 0
    while True:
        # periodically loop through all arrangements of maskings
        mu = (i // 4) % 2
        off = i % 4

        link_mask = make_schwinger_link_mask(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_schwinger_plaq_mask(
            plaq_mask_shape, mu, off, float_dtype=float_dtype, device=device
        )
        mask_2x1 = torch.zeros((2,) + tuple(plaq_mask_shape)).to(device)
        mask_2x1[1 - mu] = plaq_mask["frozen"]

        yield link_mask, (plaq_mask, {"frozen": mask_2x1})
        i += 1
