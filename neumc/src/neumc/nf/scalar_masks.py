import torch
from typing import Generator, Iterable, Iterator, Sequence
import itertools


def make_checker_mask(shape: Sequence[int], parity: int, device) -> torch.Tensor:
    """
    Make a checkerboard mask with the given shape and parity.

    Parameters
    ----------
    shape : tuple
        Dimensions of the mask.
    parity: int
        Parity of the mask. If zero mask[0,0] = 0, if one mask[0,0] = 1.
    device:
        Device on which the mask should be created.

    Returns
    -------
    torch.Tensor
        Tensor representing the mask.
    """
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(device)


def checkerboard_masks_gen(
    lattice_shape: Sequence[int], device
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    """
    Generate checkerboard masks for the lattice. It generates masks with alternating parity.

    Parameters
    ----------
    lattice_shape : tuple
        Shape of the lattice.
    device :
        Device on which the masks should be created.

    Yields
    ------
    torch.Tensor
        Checkerboard mask.
    """
    i = 0
    while True:
        parity = i % 2
        frozen_mask = make_checker_mask(lattice_shape, parity, device)
        active_mask = 1 - frozen_mask
        passive_mask = 1 - active_mask - frozen_mask
        yield ({"active": active_mask, "frozen": frozen_mask, "passive": passive_mask},)
        i += 1


def make_single_stripes(
    shape: Sequence[int],
    *,
    mu: int,
    offset: int,
    stride: int | None,
    device,
) -> torch.Tensor:
    """
    ::

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1s is repeated every `stride` row/columns.
    The pattern is offset in perpendicular to the mu direction by `offset` (mod `stride`).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = torch.zeros(shape).to(dtype=torch.uint8, device=device)
    if mu == 0:
        mask[:, 0::stride] = 1
    elif mu == 1:
        mask[0::stride] = 1
    mask = torch.roll(mask, offset, dims=1 - mu)
    return mask


def stripe_masks_gen(
    lattice_shape: Sequence[int], *, device="cpu", stride: int = 2
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    i_off = 0
    while True:
        off = i_off % 2
        for mu in range(stride):
            mask = make_single_stripes(
                lattice_shape, mu=mu, offset=off, stride=stride, device=device
            )
            yield (
                {
                    "active": mask,
                    "frozen": 1 - mask,
                    "passive": torch.zeros(lattice_shape).to(
                        device=device, dtype=torch.uint8
                    ),
                },
            )
        i_off += 1


def make_tiled_mask(shape: Sequence[int], tile: torch.Tensor) -> torch.Tensor:
    repeat = []
    for i, d in enumerate(shape):
        rem = d % tile.shape[i]
        if rem != 0:
            raise ValueError(f"shape {shape} is not divisible by tile {tile.shape}")
        repeat.append(d // tile.shape[i])

    return torch.tile(tile, repeat)


def tiled_masks_gen(
    shape: Sequence[int], tiles: Iterable[torch.Tensor]
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    for tile in itertools.cycle(tiles):
        active_mask = make_tiled_mask(shape, tile)
        frozen_mask = 1 - active_mask
        passive_mask = 1 - active_mask - frozen_mask
        (
            yield (
                {"active": active_mask, "frozen": frozen_mask, "passive": passive_mask},
            )
        )


def tiled_masks_generator(
    shape: Sequence[int], tiles: Iterable[torch.Tensor]
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    return tiled_masks_gen(shape, tiles)


def gen_simple_tiles(
    shape: Sequence[int], *, device="cpu"
) -> Generator[torch.Tensor, None, None]:
    n_elem = 1
    for d in shape:
        n_elem *= d

    for i in range(n_elem):
        tile = torch.zeros((n_elem,), dtype=torch.uint8, device=device).to(
            dtype=torch.uint8
        )
        tile[i] = 1
        yield tile.view(shape)


def make_vector(gen: Iterator):
    while True:
        mask = next(gen)
        yield {
            "active": mask[0]["active"].unsqueeze(0),
            "frozen": mask[0]["frozen"].unsqueeze(0),
            "passive": mask[0]["passive"].unsqueeze(0),
        }
