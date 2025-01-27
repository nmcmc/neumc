"""
This file contains the implementation of the coupling flows as described in [1].

[1] Kobyzev I, Prince SJD, Brubaker MA. Normalizing Flows: An Introduction and Review of Current Methods.
IEEE Trans Pattern Anal Mach Intell. 2021 Nov;43(11):3964-3979.
"""

import torch

from neumc.nf.flow_abc import Flow
from typing import Callable
from typing_extensions import override


class CouplingLayer(Flow):
    def __init__(
        self,
        transform: Callable,
        conditioner: Callable,
        *mask: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.transform = transform
        self.mask = mask
        self.conditioner = conditioner

    @override
    def forward(self, *z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_frozen: list[torch.Tensor] = []
        z_active: list[torch.Tensor] = []
        z_passive: list[torch.Tensor] = []

        for layer, mask in zip(z, self.mask):
            z_frozen.append(mask["frozen"] * layer)
            z_active.append(mask["active"] * layer)
            z_passive.append(mask["passive"] * layer)

        parameters_ = self.conditioner(*z_frozen)
        x_active, log_J_ = self.transform(
            z_active,
            active_mask=self.mask[0]["active"]
            if len(self.mask) == 1
            else [mask["active"] for mask in self.mask],
            parameters=parameters_,
        )

        fx = self.mask[0]["active"] * x_active + z_passive[0] + z_frozen[0]

        return fx, log_J_

    @override
    def reverse(self, *x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_frozen: list[torch.Tensor] = []
        x_active: list[torch.Tensor] = []
        x_passive: list[torch.Tensor] = []

        for layer, mask in zip(x, self.mask):
            x_frozen.append(mask["frozen"] * layer)
            x_active.append(mask["active"] * layer)
            x_passive.append(mask["passive"] * layer)

        parameters_ = self.conditioner(x_frozen)
        z_active, log_J_ = self.transform.reverse(  # type: ignore
            x_active,
            active_mask=self.mask[0]["active"]
            if len(self.mask) == 1
            else [mask["active"] for mask in self.mask],
            parameters=parameters_,
        )

        z = self.mask[0]["active"] * z_active + x_passive[0] + x_frozen[0]

        return z, log_J_
