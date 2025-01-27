import numpy as np


def add_metrics(history, mtrcs):
    for key, val in mtrcs.items():
        history[key].append(val)


def average_metrics(history, avg_last_N_epochs, keys):
    avg = {}
    for key in keys:
        if history[key]:
            avg_val = np.mean(history[key][-avg_last_N_epochs:])
            avg[key] = avg_val
        else:
            avg[key] = np.nan

    return avg


def print_dict(dct, pre="", **kwargs):
    for key, val in dct.items():
        print(f"{pre}{key} {dct[key]:g}", **kwargs)


def dict_to_numpy(dct, keys):
    array = []
    for key in keys:
        array.append(np.asarray(dct[key]))
    return np.stack(array, 1)


def average_history(history, n):
    kernel = np.ones((n,)) / n
    n_cols = history.shape[1]
    avgs = []
    for i in range(1, n_cols):
        avgs.append(np.convolve(history[:, i], kernel, mode="valid"))
    n_avgs_rows = len(avgs[0])
    time = history[n // 2 : n // 2 + n_avgs_rows, 0]
    epochs = np.arange(n // 2, n // 2 + n_avgs_rows)
    avgs = [epochs] + [time] + avgs
    return avgs
