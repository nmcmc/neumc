# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
import torch

from neumc.physics.u1 import compute_u1_plaq, torch_mod
from neumc.nf.flow_abc import Transformation, TransformationSequence
from typing_extensions import override
from typing import Callable, Iterator, Sequence


class U1GaugeEquivCouplingLayer(Transformation):
    """A generic gauge equivariant coupling layer for U(1) gauge theory.

    Parameters
    ----------
    active_links_mask
        A mask that selects which links are active. The mask is a tensor of shape (2, L, L)
    loops_function
        A function that computes loops from links. The function takes a tensor of shape (2, L, L)
        and returns a sequence of tensors, each tensor corresponding to a particular set of loops.
        This allows for adding the dependence on other loops than the plaquettes to the plaquette coupling layer.
    plaq_coupling
        A coupling layer that operates on plaquettes and loops. It returns a transformed plaquettes and
        the log of the Jacobian of this transformation.

    Notes
    -----
    The layer operates on the links and transforms them in a gauge equivariant way. The transformation is done by first
    computing the plaquettes and optionally additional loops from the links. The plaquettes are then transformed by the
    plaquette coupling layer. The transformed plaquettes are then used to compute the transformed links.


    References
    ----------
    [1] M. S. Albergo, M. Dalmonte, B. N. Keller, and P. Silvi, "Gauge equivariant normalizing flows: an application to  U(1) lattice gauge theory", arXiv:2101.08176

    """

    def __init__(
        self,
        *,
        active_links_mask: torch.Tensor,
        loops_function: Callable[[torch.Tensor], Sequence[torch.Tensor]] | None = None,
        plaq_coupling: Transformation,
    ):
        super().__init__()

        self.active_links_mask = active_links_mask
        self.plaq_coupling = plaq_coupling
        self.loops_function = loops_function

    @override
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the coupling layer.

        Parameters
        ----------
        x:
            A tensor of shape (2, L, L) containing the links.

        Returns
        -------
            Transformed links and the log of the Jacobian of the transformation.
        """
        assert torch.all(z < 2 * torch.pi)
        # Compute plaquettes from the links
        plaq = compute_u1_plaq(z, mu=0, nu=1)
        # Compute additional loops if needed
        if self.loops_function:
            loops = self.loops_function(z)
        else:
            loops = []

        # Transform the plaquettes
        try:
            new_plaq, logJ = self.plaq_coupling([plaq] + loops)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError

        delta_plaq = new_plaq - plaq
        delta_links = torch.stack(
            (delta_plaq, -delta_plaq), dim=1
        )  # signs for U vs Udagger
        fx = (
            self.active_links_mask * torch_mod(delta_links + z)
            + (1 - self.active_links_mask) * z
        )

        return fx, logJ

    @override
    def reverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse pass of the coupling layer.


        Parameters
        ----------
        fx
            A tensor of shape (2, L, L) containing links.
        Returns
        -------
             Links transformed by the inverse of the forward() transformation and the
             log of the Jacobian of the transformation.
        """
        assert torch.all(x < 2 * torch.pi)
        new_plaq = compute_u1_plaq(x, mu=0, nu=1)
        if self.loops_function:
            loops = self.loops_function(x)
        else:
            loops = []
        plaq, logJ = self.plaq_coupling.reverse(
            [
                new_plaq,
            ]
            + loops
        )
        delta_plaq = plaq - new_plaq
        delta_links = torch.stack(
            (delta_plaq, -delta_plaq), dim=1
        )  # signs for U vs Udagger
        x = (
            self.active_links_mask * torch_mod(delta_links + x)
            + (1 - self.active_links_mask) * x
        )

        return x, logJ


def make_u1_equiv_layers(
    *,
    make_plaq_coupling: Callable[..., Transformation],
    masks: Iterator[tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]],
    n_layers: int,
    device,
    loops_function: Callable[[torch.Tensor], Sequence[torch.Tensor]] | None = None,
):
    """
    Make a list of `n_layers` of `U1GaugeEquivCouplingLayer` coupling layers.

    Parameters
    ----------
    make_plaq_coupling
        a function that makes a plaquette coupling layer with given mask
    masks
        a list or generator of masks. Those masks will be used successively to create the n_layers coupling layers.
        Eeach item of the list is a tuple.
        The first element is the active links mask.
        The second element is the plaquette mask, which in turn is another list/tuple of masks.
        The first element of this tuple is a dictionary with keys "frozen", "active", "passive" elements.
        The rest of plaquette masks are optional and can be used for additional optional loops.
        They are always frozen masks.
    n_layers
        number of coupling layers
    device
        device to put the coupling layers on
    loops_function
        function that computes optional additional loops from links.
        The function takes a tensor of shape (N,2, L, L) representing  the links
        and returns a sequence of tensors of shape (N, d, L, L), each tensor corresponding
        to a particular set of loops. Default is None.

    Returns
    -------
    torch.nn.ModuleList
        List of coupling layers
    """

    layers = []
    for i in range(n_layers):
        link_mask, plaq_mask = next(masks)
        plaq_coupling = make_plaq_coupling(plaq_mask)
        link_coupling = U1GaugeEquivCouplingLayer(
            loops_function=loops_function,
            active_links_mask=link_mask,
            plaq_coupling=plaq_coupling,
        )

        layers.append(link_coupling)
    return TransformationSequence(layers).to(device)
