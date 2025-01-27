import torch

from torch.amp import autocast
from abc import ABC, abstractmethod
from neumc.nf.flow_abc import Transformation
from typing_extensions import override
from typing import Callable, Concatenate

import neumc.nf.flow as nf
from neumc.utils.utils import dkl


class GradientEstimator(ABC):
    """
    The abstract base class for an estimation of the gradient of the loss function.
    It is an estimation in the sense that the true loss function, i.e. usually DKL, is not necessarily calculated.
    Instead, a surrogate "loss" function is used, which then can be differentiated to estimate the gradient of the true loss function.

    The gradient is estimated according to the implementation of the `forward_pass` method. It should take a batch of samples `z` and the log probability of the samples `log_prob_z` as input.
    By default, `z` should come from the `prior` distribution and `log_prob_z` should be the log probability of the samples under the `prior` distribution.
    The forward_pass method should return the "loss",
    i.e. the value you need to call `backward` on. It should also return the log probabilities of the model (`log_q`) and the target distribution (`log_p`), respectively.

    The `step` method is the main method to call to perform a step of the optimization. It will call the forward_pass method `n_batches` times and accumulate the gradients.
    Note that the gradients are not applied to the model, it is up to the user to call the `optimizer.step()` method after the step method has been called,
    as well as to zero the gradients of the optimizer.

    When implementing a new gradient step, you should extend this class and implement the `forward_pass` method.

    Parameters
    ----------

    prior
        The prior distribution to sample from
    flow
        The normalizing flow model
    action
        The action to evaluate the log probability of the target distribution
    use_amp
        Whether to use automatic mixed precision or not
    """

    def __init__(
        self,
        prior,
        flow: Transformation,
        action: Callable[Concatenate[torch.Tensor, ...], torch.Tensor],
        use_amp: bool = False,
    ):
        self.prior = prior
        self.flow = flow
        self.use_amp = use_amp
        self.action = action

    @abstractmethod
    def forward_pass(
        self, z: torch.Tensor, log_prob_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def step(
        self, batch_size: int, n_batches: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a step of the optimization. This method will call the `forward_pass` method `n_batches` times and accumulate the gradients.

        Parameters
        ----------

        batch_size
            The size of the batch to use
        n_batches
            The number of batches to acumulate the gradients over

        Returns
        -------

        accumulated_loss
            The accumulated loss over all batches

        log_q
            The log probabilities of the model

        log_p
            The log probabilities of the target distribution

        """
        logq_list = []
        logp_list = []
        accumulated_loss = 0.0

        for _ in range(n_batches):
            with autocast("cuda", enabled=self.use_amp):
                z = self.prior.sample_n(batch_size)
                log_prob_z = self.prior.log_prob(z)

            loss, log_q, log_p = self.forward_pass(z, log_prob_z)

            loss /= n_batches  # needed since loss is averaged over all batches
            loss.backward()

            accumulated_loss += loss.detach()
            logq_list.append(log_q)
            logp_list.append(log_p)
        with torch.no_grad():
            logq = torch.cat(logq_list)
            logp = torch.cat(logp_list)
        return accumulated_loss, logq, logp


class REINFORCEEstimator(GradientEstimator):
    """
    Implementation of the REINFORCE algorithm according to the paper "Training normalizing flows with computationally intensive target probability distributions"
    by P.Bialas, T. Stebel, P. Korcyl, https://doi.org/10.1016/j.cpc.2024.109094

    Parameters
    ----------
    prior
        The prior distribution to sample from
    flow
        The normalizing flow model
    action
        The action to evaluate the log probability of the target distribution
    use_amp
        Whether to use automatic mixed precision or not
    """

    @override
    def forward_pass(self, z, log_prob_z):
        with torch.no_grad():
            with autocast("cuda", enabled=self.use_amp):
                phi, log_J = self.flow(z)
                logq = log_prob_z - log_J

                logp = -self.action(phi)
                signal = logq - logp

        with autocast("cuda", enabled=self.use_amp):
            z, log_J_rev = self.flow.reverse(phi)
            prob_z = self.prior.log_prob(z)
            log_q_phi = prob_z + log_J_rev
            loss = torch.mean(log_q_phi * (signal - signal.mean()))

        return loss, logq, logp


class RTEstimator(GradientEstimator):
    """
    Implementation of the reparametrisation trick.

    Parameters
    ----------
    prior
        The prior distribution to sample from
    flow
        The normalizing flow model
    action
        The action to evaluate the log probability of the target distribution
    use_amp
        Whether to use automatic mixed precision or not
    """

    @override
    def forward_pass(self, z, log_prob_z):
        with autocast("cuda", enabled=self.use_amp):
            x, log_J = self.flow(z)
            logq = log_prob_z - log_J

            logp = -self.action(x)
            loss = dkl(logp, logq)

        return loss, logq.detach(), logp.detach()


class PathGradientEstimator(GradientEstimator):
    """
    Implementation of the path gradient estimator. Note that log_prob_z is not used.

    Parameters
    ----------
    prior
        The prior distribution to sample from
    flow
        The normalizing flow model
    action
        The action to evaluate the log probability of the target distribution
    use_amp
        Whether to use automatic mixed precision or not
    """

    @override
    def forward_pass(self, z, log_prob_z=None):
        nf.detach(self.flow)
        with torch.no_grad():
            fi, _ = self.flow(z)
        fi.requires_grad_(True)
        zp, log_J_rev = self.flow.reverse(fi)
        prob_zp = self.prior.log_prob(zp)
        log_q = prob_zp + log_J_rev
        log_q.backward(torch.ones_like(log_J_rev))
        G = fi.grad.data
        nf.attach(self.flow)
        fi2, _ = self.flow(z)
        log_p = -self.action(fi2)
        axes = tuple(range(1, len(G.shape)))
        contr = torch.sum(fi2 * G, dim=axes)
        loss = torch.mean(contr - log_p)
        return loss, log_q.detach(), log_p.detach()
