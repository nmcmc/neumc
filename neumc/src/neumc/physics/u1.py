import numpy as np
import torch

scipy_installed = True
try:
    from scipy.special import iv
except ImportError as e:
    scipy_installed = False
    print(f"scipy is not installed: {e}")


M_2PI = 2 * torch.pi


def torch_mod(x):
    x = torch.remainder(x, M_2PI)
    x = torch.where(x >= M_2PI, x - M_2PI, x)
    return x


def torch_wrap(x):
    return torch_mod(x + np.pi) - np.pi


debug_info = {}


if scipy_installed:

    def logZ(L, beta, *, n=2):
        z = L * L * np.log(iv(0, beta))
        x = np.sum(2 * np.power(iv(np.arange(1, n + 1), beta) / iv(0, beta), L * L))
        return z + x - x * x / 2


def set_weights(m):
    if hasattr(m, "weight") and m.weight is not None:
        torch.nn.init.normal_(m.weight, mean=1, std=2)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.fill_(-1)


def compute_u1_plaq(links, mu, nu):
    """Compute U(1) plaquettes in the (mu,nu) plane given `links` = arg(U)"""
    return torch_mod(
        links[:, mu]
        + torch.roll(links[:, nu], -1, mu + 1)
        - torch.roll(links[:, mu], -1, nu + 1)
        - links[:, nu]
        + 2 * torch.pi
    )


def u1_2x1_loops(links, mu, nu):
    return torch_mod(
        links[:, mu]
        + torch.roll(links[:, mu], -1, mu + 1)
        + torch.roll(links[:, nu], -2, mu + 1)
        - torch.roll(torch.roll(links[:, mu], -1, nu + 1), -1, mu + 1)
        - torch.roll(links[:, mu], -1, nu + 1)
        - links[:, nu]
        + 2 * torch.pi
    )


def compute_u1_2x1_loops(links):
    return torch.stack((u1_2x1_loops(links, 0, 1), u1_2x1_loops(links, 1, 0)), 1)


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                action_density = action_density + torch.cos(
                    compute_u1_plaq(cfgs, mu, nu)
                )
        return -self.beta * torch.sum(action_density, dim=tuple(range(1, Nd + 1)))


def gauge_transform(links, alpha):
    transformed_links = links.clone()
    for mu in range(len(links.shape[2:])):
        transformed_links[:, mu] = torch_mod(
            alpha + links[:, mu] - torch.roll(alpha, -1, mu + 1)
        )
    return transformed_links


def random_gauge_transform(x, device):
    nconf, vol_shape = x.shape[0], x.shape[2:]
    return gauge_transform(
        x, 2 * np.pi * torch.rand((nconf,) + vol_shape, device=device)
    )


def topo_charge(x):
    P01 = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(P01.shape)))
    return torch.sum(P01, dim=axes) / (2 * np.pi)
