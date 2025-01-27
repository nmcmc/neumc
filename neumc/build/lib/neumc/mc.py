import torch
import numpy as np


def metropolize(samples_q, log_q, log_p):
    """
    Produces a Markov chain of samples from a proposal distribution q by accepting or rejecting the samples based on the
    ratio of the target distribution p and the proposal distribution q.

    Parameters
    ----------
    samples_q
        samples from distribution q
    log_q
        log of the proposal distribution q on the samples
    log_p
        log of the target distribution p on the samples
    Returns
    -------
    samples_p
        samples from distribution p
    samples_p_log_q
        log of the proposal distribution q on the samples from distribution p
    samples_p_log_p
        log of the target distribution p on the samples from distribution p
    accepted
        array of 0s and 1s indicating whether the sample was accepted
    """
    samples_p = samples_q.clone()
    samples_p_log_q = log_q.clone()
    samples_p_log_p = log_p.clone()

    r = torch.rand(len(samples_p), device=samples_q.device)

    log_w = log_p - log_q
    prev_log_w = log_w[0]
    accepted = torch.zeros(len(samples_p))
    for i in range(1, len(samples_p)):
        log_quot = log_w[i] - prev_log_w
        if r[i] < torch.exp(log_quot):
            prev_log_w = log_w[i]
            accepted[i] = 1
        else:
            samples_p[i] = samples_p[i - 1]
            samples_p_log_q = log_q[i - 1]
            samples_p_log_p = log_p[i - 1]

    return samples_p, samples_p_log_q, samples_p_log_p, accepted


def metropolize_numpy(samples_q, log_q, log_p):
    samples_p = samples_q.copy()
    r = np.random.rand(len(samples_p))

    log_w = log_p - log_q
    prev_log_w = log_w[0]
    accepted = np.zeros(len(samples_p))
    for i in range(1, len(samples_p)):
        log_quot = log_w[i] - prev_log_w
        if r[i] < np.exp(log_quot):
            prev_log_w = log_w[i]
            accepted[i] = 1
        else:
            samples_p[i] = samples_p[i - 1]

    return samples_p, accepted
