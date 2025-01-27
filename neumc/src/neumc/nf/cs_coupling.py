import torch
import numpy as np

from neumc.nf.coupling_flow import CouplingLayer
from neumc.physics.u1 import torch_mod
import neumc.splines.cs as cs
from neumc.nf.utils import make_sin_cos


class CSTransform:
    def __init__(self):
        super().__init__()

    @staticmethod
    def _call(
            x,
            *,
            dir: int,
            active_mask: torch.Tensor,
            parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        w, h, d, t = parameters
        device = w.device
        kx, ky, s = cs.make_circular_knots_array(w, h, d, device=device)
        idx = cs.make_idx(*kx.shape[:-1], device=device)
        spline = cs.make_splines_array(kx, ky, s, idx)[dir]

        x1 = active_mask * torch_mod(x - dir * t)

        fx, local_logJ = spline(x1)
        fx += (1 - dir) * t

        local_logJ = active_mask * local_logJ
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)

        return torch_mod(fx), logJ

    @staticmethod
    def forward(
            x: torch.Tensor,
            *,
            active_mask: torch.Tensor,
            parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        return CSTransform._call(
            x, dir=0, active_mask=active_mask, parameters=parameters
        )

    @staticmethod
    def reverse(
            x: torch.Tensor,
            *,
            active_mask: torch.Tensor,
            parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        return CSTransform._call(
            x, dir=1, active_mask=active_mask, parameters=parameters
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


cs_coupling = CSTransform()


class CSConditioner(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, n_knots: int):
        super().__init__()
        self.net = net
        self.n_bins = n_knots - 1

    def forward(
            self, *xs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        net_in = torch.cat([make_sin_cos(x) for x in xs], dim=1)

        net_out = torch.permute(self.net(net_in), (0, 2, 3, 1))

        w, h, d, t = (
            net_out[..., : self.n_bins],
            net_out[..., self.n_bins: 2 * self.n_bins],
            net_out[..., 2 * self.n_bins: 3 * self.n_bins],
            net_out[..., -1],
        )

        w, h, d = (
            torch.softmax(w, -1),
            torch.softmax(h, -1),
            torch.nn.functional.softplus(d),
        )

        return w, h, d, t


class CSCoupling(CouplingLayer):
    def __init__(
            self, net: torch.nn.Module, mask: dict[str, torch.Tensor], n_knots: int
    ):
        super().__init__(
            conditioner=CSConditioner(net, n_knots), transform=cs_coupling, mask=mask
        )
