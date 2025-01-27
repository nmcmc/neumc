import torch

from neumc.utils.batch_function import batch_function
from neumc.physics.u1 import U1GaugeAction


def Hopping(u, device, *, float_dtype=torch.complex64):
    id = torch.eye(2, device=device)
    pauli = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, -1.0j], [1.0j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=torch.cfloat,
        device=device,
    )
    L = u.shape[-1]
    M = torch.zeros(
        (u.shape[0], 2 * L * L, 2 * L * L), dtype=float_dtype, device=device
    )

    for x0 in range(0, L):
        for x1 in range(0, L):
            x = L * x1 + x0
            # mu=1
            y1 = x1
            y0 = x0 + 1
            if y0 >= L:
                y0 -= L
            y = L * y1 + y0
            assert x != y
            M[:, 2 * y : 2 * y + 2, 2 * x : 2 * x + 2] = (id + pauli[1]) * torch.exp(
                -1.0j * u[:, 1, x1, x0]
            ).reshape(-1, 1, 1)
            y0 = x0 - 1
            if y0 < 0:
                y0 += L
            y = L * y1 + y0
            assert x != y
            M[:, 2 * y : 2 * y + 2, 2 * x : 2 * x + 2] = (id - pauli[1]) * torch.exp(
                1.0j * u[:, 1, y1, y0]
            ).reshape(-1, 1, 1)

            # mu=0
            sign = 1
            y0 = x0
            y1 = x1 + 1
            if y1 >= L:
                y1 -= L
                sign = -1
            y = L * y1 + y0
            assert x != y
            M[:, 2 * y : 2 * y + 2, 2 * x : 2 * x + 2] = (
                sign
                * (id + pauli[0])
                * torch.exp(-1.0j * u[:, 0, x1, x0]).reshape(-1, 1, 1)
            )
            sign = 1
            y1 = x1 - 1
            if y1 < 0:
                y1 += L
                sign = -1
            y = L * y1 + y0
            assert x != y
            M[:, 2 * y : 2 * y + 2, 2 * x : 2 * x + 2] = (
                sign
                * (id - pauli[0])
                * torch.exp(1.0j * u[:, 0, y1, y0]).reshape(-1, 1, 1)
            )

    return M


def Dirac(u, kappa, device, *, float_dtype=torch.complex64):
    L = u.shape[-1]
    M = torch.zeros(
        (u.shape[0], 2 * L * L, 2 * L * L), dtype=float_dtype, device=device
    )
    M[:, range(2 * L * L), range(2 * L * L)] = 1.0

    return M - kappa * Hopping(u, device, float_dtype=float_dtype)


# %%


class U1FermionAction:
    def __init__(self, kappa):
        self.kappa = kappa

    def __call__(self, cfgs):
        M = Dirac(cfgs, self.kappa, device=cfgs.device)
        return -2 * torch.real(torch.logdet(M))


# %%


class QEDAction:
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
        self.gauge = U1GaugeAction(beta)
        self.fermionic = U1FermionAction(kappa)

    def __call__(self, cfgs):
        return self.gauge(cfgs) + self.fermionic(cfgs)


def condensate(cfgs, kappa, *, device, float_dtype=torch.complex64):
    (Lx, Ly) = cfgs.shape[-2:]
    dirac = Dirac(cfgs.to(device), kappa, device=device, float_dtype=float_dtype)
    cond = torch.diagonal(torch.linalg.inv(dirac), 0, -1, -2).sum(-1) / (Lx * Ly)
    return torch.real(cond)


def sig(cfgs, kappa, *, device, float_dtype=torch.complex64):
    dirac = Dirac(cfgs.to(device), kappa, device=device, float_dtype=float_dtype)
    return torch.cos(torch.imag(torch.logdet(dirac)))


def calc_condensate(cfgs, *, kappa, batch_size, device, float_dtype=torch.complex64):
    return batch_function(
        cfgs,
        batch_size=batch_size,
        function=lambda cfg: condensate(
            cfg, kappa, device=device, float_dtype=float_dtype
        ),
        device=device,
    )


def calc_sig(cfgs, kappa, batch_size, device, *, float_dtype=torch.complex64):
    return batch_function(
        cfgs,
        batch_size=batch_size,
        function=lambda cfg: sig(cfg, kappa, device=device, float_dtype=float_dtype),
        device=device,
    )
