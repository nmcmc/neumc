import torch
from typing import overload, TypeVar
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp as logsumexp_np


T = TypeVar("T", bound=np.number)


@overload
def dkl(logp: NDArray[T], logq: NDArray[T]) -> T: ...


@overload
def dkl(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor: ...


def dkl(logp, logq):
    return (logq - logp).mean()


@overload
def ess_lw(logw: NDArray[T]) -> T: ...


@overload
def ess_lw(logw: torch.Tensor) -> torch.Tensor: ...


def ess_lw(logw):
    """
    Compute the effective sample size given the log of importance weights.
    Parameters
    ----------
    logw
        log of importance weights

    Returns
    -------
        effective sample size
    """
    if isinstance(logw, torch.Tensor):
        lse = torch.logsumexp
        exp = torch.exp
    else:
        lse = logsumexp_np
        exp = np.exp
    log_ess = 2 * lse(logw, 0) - lse(2 * logw, 0)
    ess_per_config = exp(log_ess) / len(logw)
    return ess_per_config


@overload
def ess(logp: NDArray[T], logq: NDArray[T]) -> T: ...


@overload
def ess(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor: ...


def ess(logp, logq):
    """
    Compute the effective sample size given the log probabilities of the target and proposal distributions.

    Parameters
    ----------
    logp
        log probability of the target distribution
    logq
        log probability of the proposal distribution

    Returns
    -------
        effective sample size
    """
    logw = logp - logq
    return ess_lw(logw)
