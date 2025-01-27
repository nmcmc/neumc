from collections.abc import Sequence
from typing import Iterator
import torch

from neumc.nf.nn import make_conv_net
from neumc.nf.flow_abc import TransformationSequence
from neumc.nf.coupling_flow import CouplingLayer


class AffineTransform:
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
        x, *, active_mask: torch.Tensor, parameters: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = parameters

        fx = t + x * torch.exp(s)
        axes = range(1, len(s.size()))
        logJ = torch.sum(active_mask * s, dim=tuple(axes))

        return fx, logJ

    @staticmethod
    def reverse(
        fx, *, active_mask: torch.Tensor, parameters: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = parameters

        x = (fx - t) * torch.exp(-s)
        axes = range(1, len(s.size()))
        logJ = torch.sum(active_mask * (-s), dim=tuple(axes))

        return x, logJ

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


affine_coupling = AffineTransform()


class AffineNetConditioner(torch.nn.Module):
    """
    This is a simple adapter that applies a convolutional neural network.
    It return a tuple of output channels divided into two parts equal in size.
    Note that the last channel is lost if the number of channels in output is odd.
    """

    def __init__(self, net: torch.nn.Module):
        """
        Parameters
        ----------
        net: torch.nn.Module
            a convolutional neural network
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the convolutional neural network to the input tensor.
        The output channels of the neural network are divided into two parts
        and returned as a tuple of parameters for the affine transformation.

        Parameters
        ----------
        x: torch.Tensor
            a batch of input configurations

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            a batch of parameters for the affine transformation
        """
        out = self.net(x)
        D = out.shape[1] // 2
        return out[:, 0:D], out[:, D : 2 * D]


class AffineScalarNetConditioner(torch.nn.Module):
    """
    This is a simple adapter that applies a convolutional neural network. It unsqueezes the input tensor to add a channel
    dimension as required by the convolutional neural network. The two output channels of the neural network are
    returned as a tuple.
    """

    def __init__(self, net: torch.nn.Module):
        """
        Parameters
        ----------
        net: torch.nn.Module
            a convolutional neural network
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the convolutional neural network to the input tensor unsqueezing it firts to add a channel dimension.
        The two output channels of the neural network are returned as a tuple of parameters for the affine transformation.

        Parameters
        ----------
        x: torch.Tensor
            a batch of input configurations
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            a batch of parameters for the affine transformation
        """
        out = self.net(x.unsqueeze(1))
        return out[:, 0], out[:, 1]


class AffineCoupling(CouplingLayer):
    def __init__(
        self,
        net: torch.nn.Module,
        mask: Sequence[dict[str, torch.Tensor] | dict[str, torch.Tensor]],
    ):
        super().__init__(
            conditioner=AffineScalarNetConditioner(net),
            transform=affine_coupling,
            mask=mask,
        )


def make_scalar_affine_layers(
    *,
    lattice_shape,
    n_layers,
    masks: Iterator[Sequence[dict[str, torch.Tensor]]],
    hidden_channels,
    kernel_size,
    dilation=1,
    device,
    float_dtype=torch.float32,
    activation=torch.nn.LeakyReLU,
    bias=True,
):
    layers = []

    for i in range(n_layers):
        net = make_conv_net(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=2,
            kernel_size=kernel_size,
            use_final_tanh=True,
            float_dtype=float_dtype,
            dilation=dilation,
            activation=activation,
            bias=bias,
        )
        mask = next(masks)
        if mask[0]["active"].shape != lattice_shape:
            raise ValueError(
                f"Lattice_shape {lattice_shape} != mask.shape {mask[0]['active'].shape}"
            )
        coupling = AffineCoupling(net, mask=mask)

        layers.append(coupling)

    return TransformationSequence(layers).to(device=device)
