import torch
from torch.amp import autocast

import neumc


def make_optimizer(opt_name, opt_par, model):
    return getattr(torch.optim, opt_name)(model["layers"].parameters(), **opt_par)


def make_scheduler(scheduler_name, scheduler_par, optimizer):
    if scheduler_name:
        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer, **scheduler_par
        )


def make_loss_fn(estimator):
    return getattr(neumc.training.loss, estimator)


def step(
    batch_size, *, model, action, loss_fn, n_batches=1, use_amp
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logq_list = []
    logp_list = []
    loss_ = 0.0
    prior = model["prior"]

    for i in range(n_batches):
        with autocast("cuda", enabled=use_amp):
            z = prior.sample_n(batch_size=batch_size)
            log_prob_z = prior.log_prob(z)
        try:
            l, logq, logp = loss_fn(
                z, log_prob_z, model=model, action=action, use_amp=use_amp
            )
        except AssertionError:
            raise AssertionError

        l /= n_batches
        l.backward()

        loss_ += l.detach()
        logq_list.append(logq)
        logp_list.append(logp)

    with torch.no_grad():
        logq = torch.cat(logq_list)
        logp = torch.cat(logp_list)

    return loss_, logq, logp


def train_step(
    *,
    model,
    action,
    loss_fn,
    batch_size,
    optimizer,
    scheduler=None,
    n_batches=1,
    use_amp,
    grad_clip=None,
):
    optimizer.zero_grad(set_to_none=True)

    loss_, logq, logp = step(
        batch_size=batch_size,
        model=model,
        action=action,
        loss_fn=loss_fn,
        n_batches=n_batches,
        use_amp=use_amp,
    )

    if grad_clip is not None:
        if grad_clip.clip_value > 0.0:
            torch.nn.utils.clip_grad_value_(
                model["layers"].parameters(), clip_value=grad_clip.clip_value
            )
        if grad_clip.max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model["layers"].parameters(), max_norm=grad_clip.max_norm
            )

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        dkl = neumc.utils.dkl(logp, logq)
        std_dkl = (logq - logp).std()
        ess = neumc.utils.ess(logp, logq)
    return {
        "ess": neumc.utils.grab(ess).item(),
        "loss": loss_.cpu(),
        "dkl": neumc.utils.grab(dkl).item(),
        "std_dkl": neumc.utils.grab(std_dkl).item(),
    }
