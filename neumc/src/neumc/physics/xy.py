import torch

class XYAction:
    def __init__(self, beta=1.0) -> None:
        self.beta = beta

    def __call__(self, cfgs):
        Nd = len(cfgs.shape) - 1
        dims = range(1, Nd + 1)
        action_density = 0.0
        for mu in dims:
            action_density -= self.beta * torch.cos(cfgs - torch.roll(cfgs, -1, mu))
        return torch.sum(action_density, dim=tuple(dims))

