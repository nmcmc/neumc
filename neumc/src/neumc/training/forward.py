from typing import Callable, Concatenate, Generic, Iterable, Self, TypeVar
from neumc.nf.flow_abc import Transformation
import torch

T = TypeVar("T")


class RepeatIterator(Generic[T]):
    def __init__(self, iterable: Iterable[T]):
        self.iterable = iterable
        self.iter = iter(iterable)

    def __iter__(self) -> Self:
        self.iter = iter(self.iterable)
        return self

    def __next__(self) -> T:
        try:
            n = next(self.iter)
        except StopIteration:
            self.__iter__()
            n = next(self.iter)

        return n


class ForwardGradientEstimator:
    """
    The class that accumulate gradients using the given dataset.
    """

    def __init__(
        self,
        prior,
        flow: Transformation,
        action: Callable[Concatenate[torch.Tensor, ...], torch.Tensor],
        dataloader,
        device,
        use_amp=False,
    ):
        super().__init__()
        self.prior = prior
        self.flow = flow
        self.use_amp = use_amp
        self.action = action
        self.dataloader = dataloader
        self.data_iter = RepeatIterator(dataloader)
        self.device = device

    def forward_pass(
        self, phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, log_q_phi = self.flow.reverse(phi)

        prob_z = self.prior.log_prob(z)
        log_q_phi = prob_z + log_q_phi
        loss = -log_q_phi.mean()
        with torch.no_grad():
            log_prob_p = -self.action(phi)

        return loss, log_q_phi, log_prob_p

    def step(
        self, n_batches: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logq_list = []
        logp_list = []
        accumulated_loss = 0.0
        for _ in range(n_batches):
            (phi,) = next(self.data_iter)
            phi = phi.to(self.device)
            loss, log_q, log_p = self.forward_pass(phi)
            loss /= n_batches
            loss.backward()
        accumulated_loss += loss.detach()
        logq_list.append(log_q)
        logp_list.append(log_p)

        with torch.no_grad():
            logq = torch.cat(logq_list)
            logp = torch.cat(logp_list)
        return accumulated_loss, logq, logp
