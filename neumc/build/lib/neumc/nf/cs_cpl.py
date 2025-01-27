import torch

from neumc.physics.u1 import torch_mod
import neumc.splines.cs as cs
from neumc.nf.utils import prepare_u1_input
from neumc.nf.flow_abc import Flow
from typing_extensions import override


class CSPlaqCouplingLayer(Flow):
    """
    A plaquettes' coupling layer for U(1) gauge theory using circular rational splines.


    Parameters
    ----------
    n_knots: int
        The number of knots used in  splines.
    net: torch.nn.Module
        The neural network that produces the parameters for the rational splines
    masks
        List that contains the masks for the plaquettes and loops.
    device
        Device on which the layer should be created.


    Notes
    -----
    The neural network takes as input the (N, C, L, L) tensor where N is the batch size,
    L is the lattice size, and C is the number of input channels.

    The number of input channels depends on the number and sizes of additional Wilson loops passed to the neural network.
    See the description of the :code:`forward` method.
    The neural network outputs a tensor of shape (N, 3*(n_knots-1)+1, L, L).
    This we permute to have the channels as the last dimension (N, L, L, 3*(n_knots-1)+1).
    The first n_knots-1 channels are the widths, the next n_knots-1 are the heights,
    and the last n_knots-1 are the derivatives. The last channel is the translation.
    The widths and the heights are passed through a softmax function to ensure that they are positive and sum to one.
    They are then used to create the knots.


    References
    ----------
    Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural Spline Flows.
    https://doi.org/10.48550/arxiv.1906.04032

    Rezende, D. J., Papamakarios, G., Racanière, S., Albergo, M. S., Kanwar, G., Shanahan, P. E., & Cranmer,
    K. (2020). Normalizing Flows on Tori and Spheres. 37th International Conference on Machine Learning, ICML 2020,
    PartF16814, 8039–8048. https://doi.org/10.48550/arxiv.2002.02428
    """

    def __init__(self, *, n_knots, net: torch.nn.Module, masks, device):
        super().__init__()
        self.n_knots = n_knots
        self.n_bins = n_knots - 1
        self.plaq_mask = masks[0]
        self.loop_masks = masks[1:]
        self.net = net
        self.device = device

    def _call(self, x, dir, *loops):
        """
        Transform plaquettes using the circular rational splines.


        Parameters
        ----------
        x
            The input tensor of shape (N, L, L) representing the plaquettes.
        dir
            The direction of the transformation. If 0 the forward transformation is performed, if 1 the reverse
            transformation is performed.
        loops
            List of tensors representing additional Wilson loops used as the input to the neural network.

        Returns
        -------
        torch.Tensor
            The transformed plaquettes.
        torch.Tensor
            The log of the Jacobian of the transformation for each configuration in the batch.

        """
        net_in = prepare_u1_input(x, self.plaq_mask, loops, self.loop_masks)

        # The convolutional neural network outputs a tensor of shape (N, C_out, L, L) where C_out is the number
        # of output channels. We permute the tensor to have the channels as the last dimension.
        net_out = torch.permute(self.net(net_in), (0, 2, 3, 1))

        w, h, d, t = (
            net_out[..., : self.n_bins],
            net_out[..., self.n_bins : 2 * self.n_bins],
            net_out[..., 2 * self.n_bins : 3 * self.n_bins],
            net_out[..., -1],
        )

        w, h, d = (
            torch.softmax(w, -1),
            torch.softmax(h, -1),
            torch.nn.functional.softplus(d),
        )

        kx, ky, s = cs.make_circular_knots_array(w, h, d, device=self.device)
        idx = cs.make_idx(*kx.shape[:-1], device=self.device)
        spline = cs.make_splines_array(kx, ky, s, idx)[dir]

        x1 = self.plaq_mask["active"] * torch_mod(x - dir * t)

        fx1, local_logJ = spline(x1)
        fx1 += (1 - dir) * t

        fx1 = self.plaq_mask["active"] * fx1
        local_logJ = self.plaq_mask["active"] * local_logJ
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)

        fx = (
            self.plaq_mask["active"] * torch_mod(fx1)
            + self.plaq_mask["passive"] * x
            + self.plaq_mask["frozen"] * x
        )

        return fx, logJ

    @override
    def forward(self, x: torch.Tensor, *loops) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass through the  coupling layer.

        Parameters
        ----------
        x
            The input tensor of shape (N, L, L) representing the plaquettes.
        loops
            List of tensors representing additional Wilson loops used as the input to the neural network.


        Returns
        -------
        torch.Tensor
            The transformed plaquettes.
        torch.Tensor
            The log of the Jacobian of the transformation for each configuration in the batch.

        """
        return self._call(x, 0, *loops)

    @override
    def reverse(self, x: torch.Tensor, *loops) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse pass through the coupling layer.

        For parameters see the :code:`forward` method.

        """
        return self._call(x, 1, *loops)
