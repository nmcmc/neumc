# TODO: Add docstring to this file
import torch

def _make_list(k, n):
    if isinstance(k, int):
        return [k]*n
    else:
        return k

def make_conv_net(
    *,
    in_channels,
    hidden_channels,
    out_channels,
    kernel_size,
    use_final_tanh,
    dilation=1,
    activation=torch.nn.LeakyReLU,
    bias=True,
    float_dtype=torch.float32,
):
    """
    Create a convolutional neural network with given activations and a optional final Tanh activation.

    Parameters
    ----------
    in_channels
    hidden_channels
    out_channels
    kernel_size
    use_final_tanh
    dilation
    float_dtype

    Returns
    -------

    """
    sizes = [in_channels] + hidden_channels + [out_channels]
    net = []
    dilations = _make_list(dilation, len(sizes)-1)
    kernel_sizes = _make_list(kernel_size, len(sizes)-1)
    for i in range(len(sizes) - 1):
        net.append(
            torch.nn.Conv2d(
                sizes[i],
                sizes[i + 1],
                kernel_sizes[i],
                padding=dilations[i] * (kernel_size - 1) // 2,
                stride=1,
                padding_mode="circular",
                dilation=dilations[i],
                dtype=float_dtype,
                bias=bias,
            )
        )
        if i != len(sizes) - 2:
            net.append(activation())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)
