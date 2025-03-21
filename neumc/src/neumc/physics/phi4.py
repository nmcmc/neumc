import numpy as np
import torch


class ScalarPhi4Action:
    def __init__(self, m2, lam, kappa=1.0) -> None:
        self.m2 = m2
        self.lam = lam
        self.kappa = kappa
        self.inv_24 = 1.0 / 24.0

    def __call__(self, cfgs):
        Nd = len(cfgs.shape) - 1
        action_density = (
                                 0.5 * self.m2 + Nd * self.kappa
                         ) * cfgs ** 2 + self.inv_24 * self.lam * cfgs ** 4

        dims = range(1, Nd + 1)
        for mu in dims:
            action_density -= self.kappa * cfgs * torch.roll(cfgs, -1, mu)
        return torch.sum(action_density, dim=tuple(dims))


def phi2(L, m2):
    q = np.arange(0, L)
    q_0, q_1 = np.meshgrid(q, q)
    k2_0 = 4 * np.sin(np.pi / L * q_0) ** 2
    k2_1 = 4 * np.sin(np.pi / L * q_1) ** 2

    return np.sum(1 / (k2_0 + k2_1 + m2))


def free_field_free_energy(L, m2):
    """
    Calculates the free energy of a free scalar field in 2D with mass squared M2 on a LxL lattice.

    :param L: length of the lattice edge.
    :param m2: float mass squared
    :return: free-field free energy
    """
    q = np.arange(0, L)
    q_0, q_1 = np.meshgrid(q, q)
    k2_0 = 4 * np.sin(np.pi / L * q_0) ** 2
    k2_1 = 4 * np.sin(np.pi / L * q_1) ** 2

    return -0.5 * (L * L) * np.log(2 * np.pi) + 0.5 * np.log(k2_0 + k2_1 + m2).sum()


def mag2(cfgs):
    vol = torch.prod(torch.tensor(cfgs.shape[1:]))
    return torch.mean(cfgs.sum(dim=tuple(range(1, len(cfgs.shape)))) ** 2 / vol)


class VectorPhi4Action:
    def __init__(self, m2, lam, kappa=1.0) -> None:
        self.m2 = m2
        self.lam = lam
        self.kappa = kappa
        self.inv_24 = 1.0 / 24.0

    def __call__(self, cfgs):
        Nd = len(cfgs.shape) - 2  # number of spatial dimensions
        cfgs2 = torch.sum(cfgs ** 2, 1)
        action_density = (
                                 0.5 * self.m2 + Nd * self.kappa
                         ) * cfgs2 + self.inv_24 * self.lam * cfgs2 ** 2
        dims = range(1, Nd + 1)
        action = torch.sum(action_density, dim=tuple(dims))
        #print(f"Action shape = {action.shape}")

        mu_dims = range(2, Nd + 2)
        kinetic = 0.0
        for mu in mu_dims:
            kinetic -= cfgs * torch.roll(cfgs, -1, mu)
        #print(f"Kinetic shape = {kinetic.shape}")
        return action + self.kappa * torch.sum(kinetic, dim=tuple(range(1, Nd + 2)))
