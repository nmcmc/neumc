"""
Utility functions to generate masks for U1 lattice gauge theory.
"""

from collections.abc import Sequence
from typing import Generator

import torch

import neumc


def make_2d_link_active_stripes(
        shape: Sequence[int], *, mu: int, offset: int, float_dtype,
        device) -> torch.Tensor:
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

    mask = torch.zeros(*shape, device=device, dtype=float_dtype)
    mask[mu] = neumc.nf.scalar_masks.make_single_stripes(shape[1:], mu=mu, offset=offset, stride=4,
                                                         device=device)

    return mask


def make_u1_plaq_masks(

        shape: Sequence[int], *, mu: int, offset: int, device
) -> dict[str, torch.Tensor]:
    """
    Make masks for plaquette coupling layer.

    Parameters
    ----------
    shape
        latice shape
    mu
        direction of stripes in the mask
    offset
        offset of stripes
    device
        device to store the mask

    Returns
    -------
    mask
        dictionary with keys "frozen", "active", "passive" for three different kinds of masks
    """
    mask = {}
    mask["frozen"] = neumc.nf.scalar_masks.make_double_stripes(shape, mu=mu, offset=offset + 1, stride=4, device=device)
    mask["active"] = neumc.nf.scalar_masks.make_single_stripes(shape, mu=mu, offset=offset, stride=4, device=device, )
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def u1_masks_gen(
        *, lattice_shape: Sequence[int], float_dtype, device
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor]]], None, None]:
    """

    Parameters
    ----------

    lattice_shape

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
            link_mask_shape, mu=mu, offset=off, float_dtype=float_dtype, device=device
        )

        plaq_mask = make_u1_plaq_masks(lattice_shape, mu=mu, offset=off, device=device)

        yield link_mask, (plaq_mask,)
        i += 1


def make_sch_plaq_mask(
        shape: Sequence[int], mu: int, offset: int, float_dtype, device
) -> dict[str, torch.Tensor]:
    mask = {}
    mask["frozen"] = neumc.nf.scalar_masks.make_shifted_rows_mask(
        shape, mu=mu, period=2, row_offsets=[0], offset=offset + 1, float_dtype=float_dtype, device=device
    )
    mask["active"] = neumc.nf.scalar_masks.make_shifted_rows_mask(
        shape, mu=mu, period=4, row_offsets=[0, 2], offset=offset, float_dtype=float_dtype, device=device
    )
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def make_sch_link_mask(
        shape: Sequence[int], mu: int, offset: int, float_dtype, device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device)
    mask[mu] = neumc.nf.scalar_masks.make_shifted_rows_mask(shape[1:], mu=mu, period=4, row_offsets=[0, 2],
                                                            offset=offset, float_dtype=float_dtype,
                                                            device=device)

    return mask


def sch_masks_gen(
        *,
        lattice_shape: Sequence[int],
        float_dtype,
        device,
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor]]], None, None]:
    i = 0
    while True:
        # periodically loop through all arrangements of maskings
        mu = (i // 4) % 2
        off = i % 4
        link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)
        link_mask = make_sch_link_mask(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_sch_plaq_mask(
            lattice_shape, mu, off, float_dtype=float_dtype, device=device
        )

        yield link_mask, (plaq_mask,)
        i += 1


def sch_2x1_masks_gen(
        *,
        lattice_shape: Sequence[int],
        float_dtype,
        device,
) -> Generator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]], None, None]:
    i = 0
    while True:
        mu = (i // 4) % 2
        off = i % 4
        link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)
        link_mask = make_sch_link_mask(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_sch_plaq_mask(
            lattice_shape, mu, off, float_dtype=float_dtype, device=device
        )
        mask_2x1 = torch.zeros((2,) + tuple(lattice_shape)).to(device)
        mask_2x1[1 - mu] = plaq_mask["frozen"]

        yield link_mask, (plaq_mask, {"frozen": mask_2x1})
        i += 1
