import numpy as np
import torch
from scipy.special import logsumexp as logsumexp_np
from neumc.utils import ess_lw


def normalize_logw(logw):
    return logw - torch.logsumexp(logw, dim=0) + np.log(len(logw))


def nis(o_samples, logq, logp):
    logw = logp - logq
    logw = logw - logw.max()

    w = torch.exp(logw)
    w = w / torch.exp(torch.logsumexp(logw, dim=0))

    return (w * o_samples).sum()


def nis_lw_np(o_samples, logw):
    logw = logw - logw.max()

    w = np.exp(logw)
    w = w / np.exp(logsumexp_np(logw, axis=0))

    return (w * o_samples).sum()


def nis_np(o_samples, logq, logp):
    logw = logp - logq
    return nis_lw_np(o_samples, logw)


def cov_1_2_p(x, y, lw):
    dx = x - x.mean()
    dy = y - y.mean()
    return nis_lw_np(dx * dy * dy, lw)


def nis_error_np(o, lw):
    n = len(o)
    o_bar = nis_lw_np(o, lw)
    o2_bar = nis_lw_np(o * o, lw)
    var_o = o2_bar - o_bar**2
    t1 = var_o / ess_lw(lw)
    w = np.exp(lw)
    t2 = cov_1_2_p(w, o, lw) / w.mean()
    return np.sqrt((t1 + t2) / n), t1, t2


def nis_lw_with_err_np(o_samples, logw):
    o = nis_lw_np(o_samples, logw)
    o2 = nis_lw_np(o_samples**2, logw)

    err, t1, t2 = nis_error_np(o_samples, logw)
    return o, err, t2 / t1
