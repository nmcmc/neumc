from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Iterable
import torch


class Transformation(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    @override
    def forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms a batch of input configurations.

        Parameters
        ----------
        z: torch.Tensor
            configurations to transform

        Returns
        -------
        x: torch.Tensor
            Transformed configurations
        log_J: torch.Tensor
            the log of the Jacobian determinant of the transformation
        """
        ...

    def reverse(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x

        Returns
        -------
        z: torch.Tensor
            Transformed configurations
        loog_J: torch.Tensor
            the log of the Jacobian determinant of the transformation
        """
        return NotImplemented

    def sample(self, prior, batch_size: int):
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
        x, log_J = self.forward(z)
        return x, log_prob_z - log_J


class TransformationSequence(Transformation):
    def __init__(self, layers: Iterable[Transformation]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    @override
    def forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        log_J = torch.zeros(z.shape[0], device=z.device)

        for layer in self.layers:
            z, log_J_layer = layer.forward(z)
            log_J += log_J_layer

        return z, log_J

    @override
    def reverse(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        log_J = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            x, log_J_layer = layer.reverse(x)
            log_J += log_J_layer

        return x, log_J
