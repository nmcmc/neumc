# TODO: Add docstring to this file
import torch


def make_conv_net(
    *,
    in_channels,
    hidden_channels,
    out_channels,
    kernel_size,
    use_final_tanh,
    dilation=1,
    float_dtype=torch.float32,
):
    """
    Create a convolutional neural network with LeakyReLU activations and a optional final Tanh activation.

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
    if isinstance(dilation, int):
        dilations = [dilation] * (len(hidden_channels) + 1)
    else:
        dilations = dilation
    for i in range(len(sizes) - 1):
        net.append(
            torch.nn.Conv2d(
                sizes[i],
                sizes[i + 1],
                kernel_size,
                padding=dilations[i] * (kernel_size - 1) // 2,
                stride=1,
                padding_mode="circular",
                dilation=dilations[i],
                dtype=float_dtype,
            )
        )
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)
