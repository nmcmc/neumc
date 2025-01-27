"""
This module implements the rational spline flow.
"""

import torch


def make_idx(*dims, device="cpu"):
    r"""
    This function creates the indices needed for smart indexing.


    Parameters
    ----------
    dims
        dimension of the tensor for which the indices are created.
    Returns
    -------
        Tuple of tensors of shape :math:`(*dims)`.

    Notes
    -----
    Smart indexing in PyTorch (and numpy) works as follows: if we have a  tensor :math:`a` of shape :math:`(N, M, K,L)`
    and four index tensors :math:`(idx_0,idx_1,idx_2,idx_3)`  of the same shape then

    .. math::

        a[idx_0, idx_1, idx_2, idx_3][i,j,k,l] = a[idx_0[i,j,k,l], idx_1[i,j,k,l], \\
                                             idx_2[i,j,k,l], idx_3[i,j,k,l]]

    If we want to create a tensor such that

    .. math::

        a[i, j, k, l] = x[i, j, k, idx_3[i, j, k, l]]

    then we need to create a index tensors :math:`idx_i` as follows

    .. math::

        idx_0[i,j,k,l] &= i\\
        idx_1[i,j,k,l] &= j\\
        idx_2[i,j,k,l] &= k

    This is implemented in this function using the :code:`torch.arange` and the :code:`torch.expand` functions


    """
    idx = []
    for i, d in enumerate(dims):
        v = torch.ones(len(dims)).to(dtype=torch.int64)
        v[i] = -1
        ix = torch.arange(d, device=device).view(*v).expand(*dims)
        idx.append(ix)
    return tuple(idx)


def make_circular_knots_array(w, h, d, device="cpu"):
    """
    Creates the knots' tensors for the circular rational splines.


    Parameters
    ----------
    w : torch.Tensor (N_b, *N_s, N_k - 1)
        The distances between the knots in the x direction.
    h : torch.Tensor (N_b, *N_s, N_k - 1)
        The distances between the knots in the y direction.
    d : torch.Tensor (N_b, *N_s, N_k - 1)
        The derivatives of the spline at the knots.
    device
        Device to use for the tensors.


    Returns
    -------
    tuple
        - x (torch.Tensor): The x coordinates of the knots such that first and last knots are at 0 and :math:`2\pi`.
        - y (torch.Tensor): The y coordinates of the knots such that first and last knots are at 0 and :math:`2\pi`.
        - s (torch.Tensor): The derivative of the spline at the knots such that the derivative at the first knot is the same as the derivative at the last knot.

    Notes
    -----
    All the input tensors are of shape :math:`(N_b, *N_s, N_k - 1)`, where :math:`N_b` is the batch size, :math:`N_k`
    is the number of knots, and :math:`N_s` is the shape of the spatial dimensions. Because we are creating a circular spline,
    the derivative at the first knot is the same as the derivative at the last knot,
    so we only need to specify :math:`N_k-1` values. All the values must be positive. For the w and h tensors,
    the sums along the second dimension must be one. The returned tensors are of shape :math:`(N_b, *N_s, N_k)`.

    """
    N_b = w.shape[0]
    N_k = w.shape[-1] + 1
    N_s = w.shape[1:-1]

    assert torch.all(w > 0)
    assert torch.all(h > 0)
    assert torch.all(d > 0)

    x = torch.empty(N_b, *N_s, N_k, device=device)

    x[..., 0] = 0.0
    x[..., 1:] = torch.cumsum(w, -1) * torch.pi * 2
    x[..., -1] = 2.0 * torch.pi

    y = torch.empty(N_b, *N_s, N_k, device=device)
    y[..., 0] = 0.0
    y[..., 1:] = torch.cumsum(h, -1) * torch.pi * 2
    y[..., -1] = 2.0 * torch.pi

    s = torch.empty(N_b, *N_s, N_k, device=device)
    s[..., 1:] = d
    s[..., 0] = s[..., -1]

    return x, y, s


def make_splines_array(x, y, d, idx):
    """
    Given the position of knots and the derivative of the splines at the knots,
    this function creates spline and inverse spline functions.


    Parameters
    ----------
    x
        x coordinates of the knots.
    y
        y coordinates of the knots.
    d
        Derivative of the spline at the knots.
    idx
        Precalculated indices for the batch and spatial dimensions. See Notes for more information.


    Returns
    -------
    spline
        function that takes a tensor of shape :math:`(N, *N_s)` and returns the value of the corresponding spline at that point and the log of the derivative.
    inverse
        function that takes a tensor of shape :math:`(N, *N_s)` and returns the value  of the corresponding inverse spline at that point and the log of the derivative.


    Notes
    -----

    The input tensors are of shape :math:`(N_b, *N_s, N_k)`, where :math:`N_b` is the batch size, :math:`N_k` is the number of knots and :math:`N_s` is the shape of the spatial dimensions.

    The :code:`torch.searchsorted` function used to find in which interval the point is located returns a tensor of
    shape :math:`(N, *N_s,1)` with the indices of the upper bound of the interval, which we squeeze to shape :math:`(
    N, *N_s)`. This index tensor is later used to  get the position of the lower and upper knots. The lower knot is
    at the index given by the index tensor minus one, and the upper knot is at the index given by the index tensor.
    To access the knot at the index given by the index tensor, we use the precalculated indices :code:`idx` as follows

    .. code-block:: python

        x[(*idx, k - 1)]
        x[(*idx, k)]

    For more explanation on the indexing, see the  :py:func:`make_idx` function.

    The function returns two functions each taking a tensor of shape :math:`(N, *N_s)`
    and returning the value of the corresponding spline at that point and the log of the derivative.

    For the formulas used in the implementation, see the paper by Durkan et al. [1].

    References
    ----------
    Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural Spline Flows.
    https://doi.org/10.48550/arxiv.1906.04032
    """

    N_b = x.shape[0]
    N_s = x.shape[1:-1]

    assert x.shape == y.shape
    assert x.shape == d.shape

    assert len(idx) == len(x.shape) - 1
    assert all([i.shape == x.shape[:-1] for i in idx])

    w = x[..., 1:] - x[..., :-1]
    s = (y[..., 1:] - y[..., :-1]) / w

    assert torch.all(w > 0)
    assert torch.all(s > 0)
    assert torch.all(d > 0)

    def gamma(k, y, s, d, xi):
        ki = (*idx, k - 1)
        kip = (*idx, k)

        a = (y[kip] - y[ki]) * (s[ki] * xi * xi + d[ki] * xi * (1 - xi))
        b = s[ki] + (d[kip] + d[ki] - 2 * s[ki]) * xi * (1 - xi)
        return y[ki] + a / b

    def der(k, y, s, d, xi):
        ki = (*idx, k - 1)
        kip = (*idx, k)
        numerator = (
            s[ki]
            * s[ki]
            * (
                d[kip] * xi * xi
                + 2 * s[ki] * xi * (1 - xi)
                + d[ki] * (1 - xi) * (1 - xi)
            )
        )

        denominator = (s[ki] + (d[kip] + d[ki] - 2 * s[ki]) * xi * (1 - xi)) ** 2

        return numerator / denominator

    def spline(x_a):
        """
        Evaluates the spline function at the given points.

        Parameters
        ----------
        x_a
            Tensor of shape :math:`(N, *N_s)` representing the points at which to evaluate the splines.
            Every point is evaluated by the separate spline function.

        Returns
        -------
        tuple
            - Tensor of shape (N, *N_s) representing the value of the spline[i,j,k] at  x[i,j,k].
            - Tensor of shape (N, *N_s) representing the log of the derivative of the spline[i,j,k]  at  x[i,j,k].
        """

        assert x_a.shape == torch.Size([N_b, *N_s])
        with torch.no_grad():
            k = torch.searchsorted(x, x_a.unsqueeze(-1), right=True).squeeze(-1)

        ki = (*idx, k - 1)
        kip = (*idx, k)

        assert torch.all(x_a >= x[ki]), "wrong bs output lower"
        assert torch.all(x_a < x[kip]), "wrong bs output upper"

        xi = (x_a - x[ki]) / w[ki]
        assert torch.all(xi >= 0)
        derivative = der(k, y, s, d, xi)

        assert not torch.any(derivative < 0)

        return gamma(k, y, s, d, xi), torch.log(derivative)

    def inverse(y_a):
        """
        Evaluates the inverse spline function at the given points.

        For description of parameters and return values, see spline function.

        """
        assert torch.all(y_a < 2 * torch.pi)
        with torch.no_grad():
            k = torch.searchsorted(y, y_a.unsqueeze(-1), right=True).squeeze()

        ki = (*idx, k - 1)
        kip = (*idx, k)

        a = (y[kip] - y[ki]) * (s[ki] - d[ki]) + (y_a - y[ki]) * (
            d[kip] + d[ki] - 2 * s[ki]
        )

        b = (y[kip] - y[ki]) * d[ki] - (y_a - y[ki]) * (d[kip] + d[ki] - 2 * s[ki])
        c = -s[ki] * (y_a - y[ki])

        xi = 2 * c / (-b - torch.sqrt(b * b - 4 * a * c))
        derivative = der(k, y, s, d, xi)
        u = xi * w[ki] + x[ki]

        return u, -torch.log(derivative)

    return spline, inverse
