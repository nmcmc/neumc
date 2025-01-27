"""
Utility functions to generate masks for U1 lattice gauge theory.
"""

from collections.abc import Sequence
import torch
import numpy as np
from typing import Generator
from .scalar_masks import make_single_stripes as mss


def make_single_stripes(
    shape: Sequence[int], mu: int, off: int, device
) -> torch.Tensor:
    """
    Mask looking like::

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1 is repeated every 4 row/columns.
    The pattern is offset in perpendicular to the mu direction by `off` (mod 4).
    """
    return mss(shape, mu=mu, offset=off, stride=4, device=device)


def make_2d_link_active_stripes(
    shape: Sequence[int], mu: int, off: int, float_dtype, torch_device
) -> torch.Tensor:
    """
    Stripes' mask looks like in the `mu` channel (mu-oriented links)::

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

    mask = torch.from_numpy(np.zeros(shape).astype(dtype=float_dtype)).to(torch_device)
    mask[mu] = make_single_stripes(shape[1:], mu, off, torch_device)

    return mask


def make_double_stripes(
    shape: Sequence[int], mu: int, off: int, device
) -> torch.Tensor:
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


def make_plaq_masks(
    shape: Sequence[int], mu: int, off: int, device
) -> dict[str, torch.Tensor]:
    """
    Make masks for plaquette coupling layer.

    Parameters
    ----------
    shape
        latice shape
    mu
        direction of stripes in the mask
    off
        offset of stripes
    device
        device to store the mask

    Returns
    -------
    mask
        dictionary with keys "frozen", "active", "passive" for three different kinds of masks
    """
    mask = {}
    mask["frozen"] = make_double_stripes(shape, mu, off + 1, device=device)
    mask["active"] = make_single_stripes(shape, mu, off, device=device)
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def u1_masks(
    *, lattice_shape: Sequence[int], float_dtype, device
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor]]], None, None]:
    """

    Parameters
    ----------
    plaq_mask_shape

    link_mask_shape

    float_dtype

    device


    Returns
    -------

    """
    i = 0
    while True:
        # periodically loop through all arrangements of masks
        mu = i % 2
        off = (i // 2) % 4
        link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)

        link_mask = make_2d_link_active_stripes(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_plaq_masks(lattice_shape, mu, off, device=device)

        yield link_mask, (plaq_mask,)
        i += 1
