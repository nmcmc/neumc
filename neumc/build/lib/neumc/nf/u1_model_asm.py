"""Collection of utility functions to make gauge equivariant normalizing flows model."""

__all__ = ["assemble_model_from_dict"]

import inspect

import torch

from neumc.nf import cs_cpl as cs
from neumc.nf import ncp as nc
from neumc.nf.sch_masks import schwinger_masks_with_2x1_loops, schwinger_masks
from neumc.nf.u1_masks import u1_masks
import neumc.nf.u1_equiv
from neumc.nf.nn import make_conv_net
from neumc.physics import u1


def _print_args(argsinfo):
    for key, val in argsinfo.locals.items():
        print(f"{key} =  {val}")


def assemble_model_from_dict(config, device, *, verbose=0):
    if verbose > 0:
        print("Assembling model")
        argsinfo = inspect.getargvalues(inspect.currentframe())
        _print_args(argsinfo)

    lattice_shape = config["lattice_shape"]
    masking = config["masking"]
    coupling = config["coupling"]
    n_layers = config["n_layers"]
    float_dtype = config["float_dtype"]
    nn = config["nn"]

    link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)
    match masking:
        case "schwinger":
            masks = schwinger_masks(
                plaq_mask_shape=lattice_shape,
                link_mask_shape=link_mask_shape,
                float_dtype=float_dtype,
                device=device,
            )
            loops_function = None
            in_channels = 2
        case "2x1":
            masks = schwinger_masks_with_2x1_loops(
                plaq_mask_shape=lattice_shape,
                link_mask_shape=link_mask_shape,
                float_dtype=float_dtype,
                device=device,
            )
            loops_function = lambda x: [u1.compute_u1_2x1_loops(x)]
            in_channels = 6
        case "u1":
            masks = u1_masks(
                lattice_shape=lattice_shape, float_dtype=float_dtype, device=device
            )
            loops_function = None
            in_channels = 2
        case _:
            raise ValueError(f"Unknown masking type {masking}")

    match coupling:
        case "cs":
            n_knots = config.get("n_knots", None)
            if n_knots is None:
                raise ValueError(
                    "n_knots must be specified for circular spline coupling"
                )

            def make_plaq_coupling(mask):
                out_channels = 3 * (n_knots - 1) + 1
                net = make_conv_net(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=nn["hidden_channels"],
                    kernel_size=nn["kernel_size"],
                    use_final_tanh=False,
                    dilation=nn["dilation"],
                )
                net.to(device)
                return cs.CSPlaqCouplingLayer(
                    n_knots=n_knots, net=net, masks=mask, device=device
                )

        case "ncp":
            n_mixture_comps = config.get("n_mixture_comps", None)
            if n_mixture_comps is None:
                raise ValueError(
                    "n_mixture_comps must be specified for non-compact projection coupling"
                )

            def make_plaq_coupling(mask):
                out_channels = n_mixture_comps + 1
                net = make_conv_net(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=nn["hidden_channels"],
                    kernel_size=nn["kernel_size"],
                    use_final_tanh=False,
                    dilation=nn["dilation"],
                )
                net.to(device)
                return nc.NCPPlaqCouplingLayer(net, mask=mask, device=device)

        case _:
            raise ValueError(f"Unknown coupling type {coupling}")

    layers = neumc.nf.u1_equiv.make_u1_equiv_layers(
        loops_function=loops_function,
        make_plaq_coupling=make_plaq_coupling,
        masks=masks,
        n_layers=n_layers,
        device=device,
    )

    u1.set_weights(layers)
    layers.to(device)  # probably redundant
    return {
        "layers": layers,
        "prior": neumc.nf.prior.MultivariateUniform(
            torch.zeros((2, *lattice_shape)),
            2 * torch.pi * torch.ones((2, *lattice_shape)),
            device=device,
        ),
    }
