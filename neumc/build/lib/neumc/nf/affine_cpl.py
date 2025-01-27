import torch

from neumc.nf.nn import make_conv_net
from neumc.nf.flow_abc import Flow, FlowSequence
from neumc.nf.coupling_flow import CouplingLayer


class AffineTransform:
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, *, active_mask, parameters):
        s, t = parameters

        fx = t + x * torch.exp(s)
        axes = range(1, len(s.size()))
        logJ = torch.sum(active_mask * s, dim=tuple(axes))

        return fx, logJ

    @staticmethod
    def reverse(fx, *, active_mask, parameters):
        s, t = parameters

        x = (fx - t) * torch.exp(-s)
        axes = range(1, len(s.size()))
        logJ = torch.sum(active_mask * (-s), dim=tuple(axes))

        return x, logJ

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


affine_coupling = AffineTransform()


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

    def forward(self, x):
        """
        Applies the convolutional neural network to the input tensor unsqueezing it firts to add a channel dimension.
        The two output channels of the neural network are returned as a tuple of parameters for the affine transformation.

        Parameters
        ----------
        x: torch.Tensor
            a batch of input configurations
        Returns
        -------
        torch.Tensor
            a batch of parameters for the affine transformation
        """
        out = self.net(x.unsqueeze(1))
        return out[:, 0], out[:, 1]


class AffineCoupling(CouplingLayer):
    def __init__(self, net, mask):
        super().__init__(affine_coupling, AffineScalarNetConditioner(net), mask)


def make_scalar_affine_layers(
    *,
    lattice_shape,
    n_layers,
    masks,
    hidden_channels,
    kernel_size,
    dilation=1,
    device,
    float_dtype=torch.float32,
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
        )
        mask = next(masks)
        if mask["active"].shape != lattice_shape:
            raise ValueError(
                f"Lattice_shape {lattice_shape} != mask.shape {mask['active'].shape}"
            )
        coupling = AffineCoupling(net, mask=mask)

        layers.append(coupling)

    return FlowSequence(layers).to(device=device)
