# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all

from typing import Callable
from neumc.nf.coupling_flow import CouplingLayer
import numpy as np
import torch

from neumc.physics.u1 import torch_mod
from neumc.nf.utils import make_sin_cos


def tan_transform(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch_mod(2 * torch.atan(torch.exp(s) * torch.tan(x / 2)))


def tan_transform_logJ(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return -torch.log(
        torch.exp(-s) * torch.cos(x / 2) ** 2 + torch.exp(s) * torch.sin(x / 2) ** 2
    )


def mixture_tan_transform(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == len(
        s.shape
    ), f"Dimension mismatch between x and s {x.shape} vs {s.shape}"
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)


def mixture_tan_transform_logJ(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == len(
        s.shape
    ), f"Dimension mismatch between x and s {x.shape} vs {s.shape}"
    return torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])


def invert_transform_bisect(
    y, *, f: Callable, tol: float, max_iter: int, a: float = 0, b: float = 2 * np.pi
):
    min_x = a * torch.ones_like(y)
    max_x = b * torch.ones_like(y)
    min_val = f(min_x)
    max_val = f(max_x)
    with torch.no_grad():
        for i in range(max_iter):
            mid_x = (min_x + max_x) / 2
            mid_val = f(mid_x)
            greater_mask = (y > mid_val).int()
            greater_mask = greater_mask.float()
            err = torch.max(torch.abs(y - mid_val))
            if err < tol:
                return mid_x
            if torch.all((mid_x == min_x) + (mid_x == max_x)):
                print(
                    "WARNING: Reached floating point precision before tolerance "
                    f"(iter {i}, err {err})"
                )
                return mid_x
            min_x = greater_mask * mid_x + (1 - greater_mask) * min_x
            min_val = greater_mask * mid_val + (1 - greater_mask) * min_val
            max_x = (1 - greater_mask) * mid_x + greater_mask * max_x
            max_val = (1 - greater_mask) * mid_val + greater_mask * max_val
        print(
            f"WARNING: Did not converge to tol {tol} in {max_iter} iters! Error was {err}"
        )
        return mid_x


class NCPTransform:
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
        x: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = parameters
        x1 = x.unsqueeze(1)
        fx = torch.mean(tan_transform(x1, s), dim=1, keepdim=True).squeeze(1)
        fx = torch_mod(fx + t)
        log_J = active_mask * mixture_tan_transform_logJ(x1, s)
        log_J = torch.sum(log_J, dim=tuple(range(1, len(log_J.shape))))

        return fx, log_J

    @staticmethod
    def reverse(x):
        return NotImplemented

    @staticmethod
    def __call__(*args, **kwargs):
        return NCPTransform.forward(*args, **kwargs)


class NCPConditioner(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.net = net

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net_in = torch.cat([make_sin_cos(x) for x in xs], dim=1)
        net_out = self.net(net_in)
        assert net_out.shape[1] >= 2
        s, t = net_out[:, :-1], net_out[:, -1]
        return s, t


class NCPPlaqCouplingLayer(CouplingLayer):
    def __init__(self, net: torch.nn.Module, mask: dict[str, torch.Tensor]):
        super().__init__(
            conditioner=NCPConditioner(net),
            transform=NCPTransform(),
            mask=mask,
        )
