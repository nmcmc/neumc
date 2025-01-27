import torch
from torch.amp import autocast

from neumc.utils.utils import dkl
import neumc.nf.flow as flow


def REINFORCE_loss(z_a, log_prob_z_a, *, model, action, use_amp):
    layers, prior = model["layers"], model["prior"]
    with torch.no_grad():
        with autocast("cuda", enabled=use_amp):
            phi, log_J = layers(z_a)
            logq = log_prob_z_a - log_J

            logp = -action(phi)
            signal = logq - logp

    with autocast("cuda", enabled=use_amp):
        z, log_J_rev = layers.reverse(phi)
        prob_z = prior.log_prob(z)
        log_q_phi = prob_z + log_J_rev
        loss = torch.mean(log_q_phi * (signal - signal.mean()))

    return loss, logq, logp


def rt_loss(z, log_prob_z, *, model, action, use_amp):
    layers = model["layers"]

    with autocast("cuda", enabled=use_amp):
        x, log_J = layers(z)
        logq = log_prob_z - log_J

        logp = -action(x)
        loss = dkl(logp, logq)

        return loss, logq.detach(), logp.detach()


def path_gradient_loss(z, log_prob_z, *, model, action, use_amp):
    layers, prior = model["layers"], model["prior"]
    flow.detach(layers)
    with torch.no_grad():
        fi, _ = layers(z)
    fi.requires_grad_(True)
    zp, log_J_rev = layers.reverse(fi)
    prob_zp = prior.log_prob(zp)
    log_q = prob_zp + log_J_rev
    log_q.backward(torch.ones_like(log_J_rev))
    G = fi.grad.data
    flow.attach(layers)
    fi2, _ = layers(z)
    log_p = -action(fi2)
    axes = tuple(range(1, len(G.shape)))
    contr = torch.sum(fi2 * G, dim=axes)
    loss = torch.mean(contr - log_p)
    return loss, log_q.detach(), log_p.detach()
