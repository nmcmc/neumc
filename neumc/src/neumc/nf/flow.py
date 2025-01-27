# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
"""Various normalizing flow utilities."""

import torch

from neumc.nf.flow_abc import Transformation


def sample(
    n_samples: int, batch_size: int, prior, layers: Transformation
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample configurations from a normalizing flow model in batches of batch_size at a time.
    Parameters
    ----------
    n_samples
        number of configurations to sample
    batch_size
        number of configurations to sample at a time
    prior
        distribution to sample prior configurations from
    layers
        normalizing flow layers

    Returns
    -------
        samples and the log probability of the samples
    """
    rem_size = n_samples
    samples = []
    log_q = []
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            x, logq = layers.sample(prior, batch_size=batch_length)

        samples.append(x.cpu())
        log_q.append(logq.cpu())

        rem_size -= batch_length

    return torch.cat(samples, 0), torch.cat(log_q, -1)


def log_prob(
    x: torch.Tensor, prior, layers: torch.nn.ModuleList | Transformation
) -> torch.Tensor:
    z, log_J_rev = layers.reverse(x)
    prob_z = prior.log_prob(z)
    return prob_z + log_J_rev


def requires_grad(model, on=True):
    """Set requires_grad attribute on all parameters of a model."""

    for p in model.parameters():
        p.requires_grad = on


def detach(model):
    """Detach all parameters of a model."""
    requires_grad(model, False)


def attach(model):
    """Attach all parameters of a model."""
    requires_grad(model, True)
