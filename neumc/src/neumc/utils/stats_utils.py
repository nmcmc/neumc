import numpy as np
import torch


def block_mean(sample, block_size):
    n_blocks = len(sample) // block_size
    if n_blocks == 0:
        raise RuntimeError("sample shorter then block_size")
    sample = sample.reshape(n_blocks, block_size, *sample.shape[1:])
    return sample.mean(1)


def ac(series, history=100):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = np.empty(history)
    for i in range(history):
        out[i] = np.mean((series[:-history] - mu) * (series[i : i - history] - mu))
    return out / var


def ac_and_tau_int(series, c=10, maxlen=200):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = [1.0]
    tau_int = 0.5
    for t in range(1, maxlen):
        cor = np.mean((series[:-t] - mu) * (series[t:] - mu)) / var
        tau_int += cor
        out.append(cor)
        if t > c * tau_int:
            break
    return tau_int, np.asarray(out)


def list_mean(data, mean=0.0):
    mean = mean
    size = 0
    for t in data:
        mean += t.sum()
        size += t.nelement()

    return mean / size


def bootstrapf(f, x, *, n_samples, binsize=1):
    """
    Bootstrap of the estimator f on x.

    The function draws n_samples bootstrap samples of size binsize from x and computes the estimator f on each of
    them. The mean and standard deviation of the estimator are returned.  The samples are produced by drawing the
    bins from x with replacement. Default binsize is one i.e., the samples are drawn from x with replacement.
    When dealing with correlated data, the bin size should be greater the autocorrelation time of the data.

    Parameters
    ----------
    f
        estimator
    x
        data
    n_samples
        number of bootstrap samples
    binsize
        size of the bins

    Returns
    -------
    mean
        mean of the estimator
    std
        standard deviation of the estimator
    """
    n_bins = len(x) // binsize
    xb = np.stack(np.array_split(x, n_bins), axis=0)

    boots = []

    for i in range(n_samples):
        bsample = np.resize(
            xb[np.random.choice(n_bins, n_bins)], (n_bins * binsize, *x.shape[1:])
        )
        boots.append(f(bsample))
    return np.mean(boots, 0), np.std(boots, 0)


def bootstrap(x, *, n_samples, binsize):
    return bootstrapf(lambda x: np.mean(x, 0), x, n_samples=n_samples, binsize=binsize)


def torch_bootstrapf(f, x, *, n_samples, binsize):
    """
    Bootstrap of the estimator f on x.

    The function  draws n_samples bootstrap samples of size binsize from x and computes the estimator f on each of
    them. The mean and standard deviation of the estimator are returned.  The samples are produced by drawing the
    bins from x with replacement. Default binsize is 1, i.e. the samples are drawn from x with replacement.
    When dealing with correlated data, the bin size should be greater the autocorrelation time of the data.

    Parameters
    ----------
    f
        estimator
    x
        data
    n_samples
        number of bootstrap samples
    binsize
        size of the bins

    Returns
    -------
    mean
        mean of the estimator
    std
        standard deviation of the estimator
    """
    n_bins = len(x) // binsize
    xb = torch.stack(torch.tensor_split(x, n_bins), dim=0)

    boots = []

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view((n_bins * binsize, *x.shape[1:]))
        boots.append(f(bsample))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)


def torch_bootstrap(obs, *, n_samples, binsize, logweights=None):
    """
    A faster version of the bootstrap function for cases when the estimator is just a mean of input data.

    The function draws n_samples bootstrap samples of size(obs) from obs with replacement. Then it calculates the mean of each sample.
    Then it calculates the mean and standard deviation of those means. If bin size is greater than 1 the obs is
    divided into binx of binsize size. Then the bins are sampled.

    Parameters
    ----------
    obs
        observable
    n_samples
        number of bootstrap samples
    binsize
        size of the bins
    logweights
        log of importance weights used for NIS. If None, no importance weights are used.

    Returns
    -------
    mean
        mean of the means of bootstrapped samples
    std
        standard deviation of the means of bootstrapped samples
    """

    n_bins = len(obs) // binsize
    xb = torch.stack(torch.tensor_split(obs, n_bins), dim=0)

    boots = []
    if logweights is not None:
        weights = torch.exp(logweights - torch.max(logweights))
        bweights = torch.stack(torch.tensor_split(weights, n_bins), dim=0)

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view((n_bins * binsize, *obs.shape[1:]))
        if logweights is not None:
            bwsample = bweights[idx].ravel()
            norm = torch.mean(bwsample)
            boots.append(torch.mean(bwsample * bsample / norm, 0))
        else:
            boots.append(torch.mean(bsample, 0))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)


def torch_bootstrapo(obs, x, *, n_samples, binsize, logweights=None):
    """
    A faster version of the bootstrap function for cases when the estimator is just a mean of an observable applied to
    each element separately.

    The function  draws n_samples bootstrap samples of size binsize from x and computes the estimator f on each of
    them. The mean and standard deviation of the estimator are returned.  The samples are produced by drawing the
    bins from x with replacement. Default binsize is 1, i.e. the samples are drawn from x with replacement.
    When dealing with correlated data, the bin size should be greater the autocorrelation time of the data.

    Parameters
    ----------
    obs
        observable to apply to the data
    x
        data
    n_samples
        number of bootstrap samples
    binsize
        size of the bins
    logweights
        log of importance weights used for NIS. If None, no importance weights are used.

    Returns
    -------
    mean
        mean of the estimator on bootstrapped samples
    std
        standard deviation of the estimator on bootstrapped samples
    """

    obs_ = obs(x)

    return torch_bootstrap(
        obs_, n_samples=n_samples, binsize=binsize, logweights=logweights
    )
