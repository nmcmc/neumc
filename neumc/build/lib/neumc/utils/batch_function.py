import torch
from typing import Callable, Concatenate
from torch._prims_common import DeviceLikeType


def batch_function(
    cfgs: torch.Tensor,
    *,
    function: Callable[Concatenate[torch.Tensor, ...], torch.Tensor],
    batch_size: int,
    device: DeviceLikeType,
    **kwargs,
) -> torch.Tensor:
    """
    Calculate a function on a set of configurations in batches of batch_size at a time.


    Parameters
    ----------
    cfgs
        input configurations
    function
        function to apply to the configurations
    batch_size
        number of configurations to process at a time
    device
        device to run the calculations on
    kwargs
        additional arguments to pass to the function

    Returns
    -------
        results of the function applied to the configurations
    """
    rem_size = len(cfgs)

    obs = []
    i = 0
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            o = function(cfgs[i : i + batch_length].to(device), **kwargs)
            obs.append(o.cpu())
            i += batch_length
            rem_size -= batch_length

    return torch.cat(obs, -1)


def batch_action(
    cfgs: torch.Tensor,
    *,
    action: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    device: DeviceLikeType,
) -> torch.Tensor:
    return batch_function(cfgs, function=action, batch_size=batch_size, device=device)
