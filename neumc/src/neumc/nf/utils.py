import torch


def make_sin_cos(x):
    """Takes a 3 or 4 dimensional tensor and returns a four dimensional tensor that contains the cos and sin of the
    input concatenated along the dimension 1.
    If the input is three dimensional then dimension 1 is inserted into the output.
    """
    if x.dim() == 3:
        return torch.stack((torch.cos(x), torch.sin(x)), dim=1)
    elif x.dim() == 4:
        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)
    else:
        raise AssertionError


def prepare_u1_input(plaq, plaq_mask, loops=(), loops_masks=()):
    """
    Takes the plaquette and loop data and concatenates them into a single tensor that can be fed into the neural network.
    The plaquette and loops data is transformed into sin and cos components and concatenated along dimension 1.
    Each component is multiplied by the corresponding mask.

    Parameters
    ----------
    plaq: torch.Tensor
        The plaquette data
    plaq_mask: dict
        A dictionary containing the masks for the plaquette data
    loops: tuple
        A tuple containing the loop data
    loops_masks: tuple
        A tuple containing the masks for the loop data

    Returns
    -------
    torch.Tensor
        The input tensor for the neural network
    """
    p2 = plaq_mask["frozen"] * plaq
    net_in = [make_sin_cos(p2)]
    for i, l in enumerate(loops):
        sc = make_sin_cos(l * loops_masks[i])
        net_in.append(sc)
    return torch.cat(net_in, dim=1)


def softmax_with_minimum(x, dim, min_value):
    p = torch.softmax(x, dim)
    n = x.size(dim)
    assert n * min_value <1
    return (1 - n * min_value) * p + min_value

