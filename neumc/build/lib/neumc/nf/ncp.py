# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all

import numpy as np
import torch

from neumc.nf.nn import make_conv_net
import neumc.nf.u1_equiv as equiv
from neumc.nf.u1_masks import u1_masks
from neumc.physics.u1 import torch_mod
from neumc.nf.sch_masks import schwinger_masks_with_2x1_loops
from neumc.physics.u1 import compute_u1_2x1_loops
from neumc.nf.utils import prepare_u1_input
from neumc.nf.flow_abc import Flow
from typing_extensions import override


def tan_transform(x, s):
    return torch_mod(2 * torch.atan(torch.exp(s) * torch.tan(x / 2)))


def tan_transform_logJ(x, s):
    return -torch.log(
        torch.exp(-s) * torch.cos(x / 2) ** 2 + torch.exp(s) * torch.sin(x / 2) ** 2
    )


def mixture_tan_transform(x, s):
    assert len(x.shape) == len(
        s.shape
    ), f"Dimension mismatch between x and s {x.shape} vs {s.shape}"
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)


def mixture_tan_transform_logJ(x, s):
    assert len(x.shape) == len(
        s.shape
    ), f"Dimension mismatch between x and s {x.shape} vs {s.shape}"
    return torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])


def invert_transform_bisect(y, *, f, tol, max_iter, a=0, b=2 * np.pi):
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


class NCPPlaqCouplingLayer(Flow):
    def __init__(self, net, *, mask, inv_prec=1e-6, inv_max_iter=1000, device):
        super().__init__()

        self.plaq_mask = mask[0]
        self.loop_masks = mask[1:]

        self.net = net
        self.inv_prec = inv_prec
        self.inv_max_iter = inv_max_iter

    @override
    def forward(self, z: torch.Tensor, *loops) -> tuple[torch.Tensor, torch.Tensor]:
        x2 = self.plaq_mask["frozen"] * z

        net_in = prepare_u1_input(z, self.plaq_mask, loops, self.loop_masks)

        net_out = self.net(net_in)
        assert net_out.shape[1] >= 2, "CNN must output n_mix (s_i) + 1 (t) channels"
        s, t = net_out[:, :-1], net_out[:, -1]

        x1 = self.plaq_mask["active"] * z
        x1 = x1.unsqueeze(1)
        local_logJ = self.plaq_mask["active"] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)
        fx1 = self.plaq_mask["active"] * mixture_tan_transform(x1, s).squeeze(1)

        fx = (
            self.plaq_mask["active"] * torch_mod(fx1 + t)
            + self.plaq_mask["passive"] * z
            + self.plaq_mask["frozen"] * z
        )
        return fx, logJ

    @override
    def reverse(self, x: torch.Tensor, *loops) -> tuple[torch.Tensor, torch.Tensor]:
        net_in = prepare_u1_input(x, self.plaq_mask, loops, self.loop_masks)
        net_out = self.net(net_in)
        assert net_out.shape[1] >= 2, "CNN must output n_mix (s_i) + 1 (t) channels"
        s, t = net_out[:, :-1], net_out[:, -1]

        x1 = torch_mod(self.plaq_mask["active"] * (x - t).unsqueeze(1))

        def transform(x):
            return self.plaq_mask["active"] * mixture_tan_transform(x, s)

        x1 = invert_transform_bisect(
            x1, f=transform, tol=self.inv_prec, max_iter=self.inv_max_iter
        )
        local_logJ = self.plaq_mask["active"] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = -torch.sum(local_logJ, dim=axes)
        x1 = x1.squeeze(1)

        x = (
            self.plaq_mask["active"] * x1
            + self.plaq_mask["passive"] * x
            + self.plaq_mask["frozen"] * fx2
        )
        return x, logJ


def make_u1_equiv_layers(
    *,
    n_layers,
    n_mixture_comps,
    lattice_shape,
    hidden_sizes,
    kernel_size,
    dilation,
    float_dtype,
    device,
):
    def make_plaq_coupling(mask):
        in_channels = 2  # x - > (cos(x), sin(x))
        out_channels = n_mixture_comps + 1  # for mixture s and t, respectively
        net = make_conv_net(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_sizes,
            kernel_size=kernel_size,
            dilation=dilation,
            use_final_tanh=False,
        )

        return NCPPlaqCouplingLayer(net, mask=mask, device=device)

    masks = u1_masks(
        lattice_shape=lattice_shape, float_dtype=float_dtype, device=device
    )

    return equiv.make_u1_equiv_layers(
        make_plaq_coupling=make_plaq_coupling,
        masks=masks,
        n_layers=n_layers,
        device=device,
    )


def make_u1_equiv_layers_2x1_loops(
    *,
    n_layers,
    n_mixture_comps,
    lattice_shape,
    hidden_sizes,
    kernel_size,
    dilation,
    float_dtype,
    device,
):
    def make_plaq_coupling(mask):
        in_channels = 6  # x - > (cos(x), sin(x))
        out_channels = n_mixture_comps + 1  # for mixture s and t, respectively
        net = make_conv_net(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_sizes,
            kernel_size=kernel_size,
            dilation=dilation,
            use_final_tanh=False,
            float_dtype=getattr(torch, float_dtype),
        )

        return NCPPlaqCouplingLayer(net, mask=mask, device=device)

    link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)

    masks = schwinger_masks_with_2x1_loops(
        plaq_mask_shape=lattice_shape,
        link_mask_shape=link_mask_shape,
        float_dtype=float_dtype,
        device=device,
    )

    return equiv.make_u1_equiv_layers(
        loops_function=lambda x: [compute_u1_2x1_loops(x)],
        make_plaq_coupling=make_plaq_coupling,
        masks=masks,
        n_layers=n_layers,
        device=device,
    )


# %%
