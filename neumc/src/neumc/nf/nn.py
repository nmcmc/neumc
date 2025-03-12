import torch


def _make_list(k, n):
    if isinstance(k, int):
        return [k] * n
    else:
        return k


def make_conv_net(
    *,
    in_channels: int,
    hidden_channels: list,
    out_channels: int,
    kernel_size: int | list,
    use_final_tanh: bool,
    dilation: int | list = 1,
    activation: type[torch.nn.Module] = torch.nn.LeakyReLU,
    bias=True,
    float_dtype=torch.float32,
):
    """
    Create a convolutional neural network with given activations and an optional final Tanh activation.

    Parameters
    ----------
    in_channels
        number of input parameters for the first network

    hidden_channels
        list of numbers of input parameters for hidden layers

    out_channels
        number of output parameters for the last layer

    kernel_size
        Size of the kernel. If one number is provided, all kernels will have the same size. If the list
        of numbers is provided, consecutive convolutional layers will have corresponding kernel sizes.
        The list should have exactly `len(hidden_channels) + 1` elements.

    use_final_tanh
        If true, 'tanh` will be used transform the output of the last layer. In the opposite case, the output
        of the last layer will be returned without any additional transformation.

    dilation
        Number of rows and columns between consecutive convolution calculations. For 1, no skipping.
        If it is a number, all layers will have the same dilation. If is is a list, consecutive layers will have
        corresponding dilations. The list should have exactly `len(hidden_channels) + 1` elements.

    activation
        Type of activation function to be used. Note that it takes a type, not an instance of a Module. Constructor of the
        activation function should not take any additional arguments. Default: `LeakyReLU`.

    bias
        If true, bias is used in all convolutional layers. Default: `True`.

    float_dtype
        Type to be used by convolution.

    Returns
    -------

    Sequential with consecutive convolutional layers.

    """
    sizes = [in_channels] + hidden_channels + [out_channels]
    net = []
    dilations = _make_list(dilation, len(sizes) - 1)
    kernel_sizes = _make_list(kernel_size, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        net.append(
            torch.nn.Conv2d(
                sizes[i],
                sizes[i + 1],
                kernel_sizes[i],
                padding=dilations[i] * (kernel_sizes[i] - 1) // 2,
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
