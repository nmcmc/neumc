"""
This file contains the implementation of the coupling flows as described in [1].

[1] Kobyzev I, Prince SJD, Brubaker MA. Normalizing Flows: An Introduction and Review of Current Methods.
IEEE Trans Pattern Anal Mach Intell. 2021 Nov;43(11):3964-3979.
"""

from collections.abc import Sequence
from typing import Callable

import torch
from typing_extensions import override

from neumc.nf.flow_abc import Transformation


class CouplingLayer(Transformation):
    def __init__(
        self,
        transform,
        conditioner: Callable,
        mask: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    ):
        super().__init__()
        self.transform = transform
        if not isinstance(mask, Sequence):
            self.mask = (mask,)
        else:
            self.mask = mask
        self.conditioner = conditioner

    def _call(
        self, dir: int, xs: Sequence[torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(xs, Sequence):
            xs = (xs,)

        # x_active = xs[0] * self.mask[0]["active"]
        x_passive = xs[0] * self.mask[0]["passive"]

        x_frozen = [
            mask["frozen"] * item_x for mask, item_x in zip(self.mask, xs, strict=True)
        ]

        parameters_ = self.conditioner(*x_frozen)
        if dir == 0:
            z_active, log_J_ = self.transform(
                xs[0], active_mask=self.mask[0]["active"], parameters=parameters_
            )
        else:
            z_active, log_J_ = self.transform.reverse(
                xs[0], active_mask=self.mask[0]["active"], parameters=parameters_
            )

        z = self.mask[0]["active"] * z_active + x_passive + x_frozen[0]

        return z, log_J_

    @override
    def forward(
        self, z: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._call(0, z)

    @override
    def reverse(
        self, x: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._call(1, x)
