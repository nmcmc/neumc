import numpy as np
import torch
from typing import Generator


def make_2d_link_active_stripes(
    shape, mu, off, float_dtype, torch_device
) -> torch.Tensor:
    """
    Stripes mask looks like in the `mu` channel (mu-oriented links)::

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction, and the pattern is offset in the nu
    direction by `off` (mod 4). The other channel is identically 0.
    """
    assert len(shape) == 2 + 1, "need to pass shape suitable for 2D gauge theory"
    assert shape[0] == len(shape[1:]), "first dim of shape must be Nd"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[mu, :, 0::4] = 1
    elif mu == 1:
        mask[mu, 0::4] = 1
    nu = 1 - mu
    mask = np.roll(mask, off, axis=nu + 1)
    return torch.from_numpy(mask.astype(float_dtype)).to(torch_device)


def make_single_stripes(shape, mu, off, device) -> torch.Tensor:
    """
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1 is repeated every 4.
    The pattern is offset in perpendicular to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
    elif mu == 1:
        mask[0::4] = 1
    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).to(device)


# %%
def make_double_stripes(shape, mu, off, device) -> torch.Tensor:
    """
    Double stripes mask looks like::

      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0

    where vertical is the `mu` direction. The pattern is offset in perpendicular
    to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
        mask[:, 1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1
    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).to(device)


# %%
def make_plaq_masks(mask_shape, mask_mu, mask_off, device) -> dict[str, torch.Tensor]:
    mask = {}
    mask["frozen"] = make_double_stripes(
        mask_shape, mask_mu, mask_off + 1, device=device
    )
    mask["active"] = make_single_stripes(mask_shape, mask_mu, mask_off, device=device)
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def make_mask(
    shape, mu, period, row_offsets, offset, float_dtype, device
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
    shape, mu, offset, float_dtype, device
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


def make_schwinger_link_mask(shape, mu, offset, float_dtype, device) -> torch.Tensor:
    mask = torch.zeros(shape, device=device)
    mask[mu] = make_mask(shape[1:], mu, 4, [0, 2], offset, float_dtype, device)

    return mask


def schwinger_masks(
    *, plaq_mask_shape, link_mask_shape, float_dtype, device
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
    *, plaq_mask_shape, link_mask_shape, float_dtype, device
) -> Generator[
    tuple[torch.Tensor, tuple[dict[str, torch.Tensor], torch.Tensor]], None, None
]:
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

        yield link_mask, (plaq_mask, mask_2x1)
        i += 1
