# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.


from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


def init_live_plot(N_era, N_epoch, *, dpi=125, figsize=(8, 4), metric="dkl"):
    fig, ax_ess = plt.subplots(dpi=dpi, figsize=figsize)
    plt.xlim(0, N_era * N_epoch)
    plt.ylim(0, 1)
    ax_ess.grid(True)
    ess_line = ax_ess.plot([0], [0], alpha=0.5, c="blue", label="ESS")  # dummy
    ax_ess.set_ylabel("ESS")

    ax_loss = ax_ess.twinx()
    ax_loss.grid(False)
    loss_line = ax_loss.plot([0], [0], alpha=0.5, c="orange", label=metric)  # dummy
    ax_loss.set_ylabel(metric)

    artists = ess_line + loss_line
    labs = [l.get_label() for l in artists]
    ax_ess.legend(artists, labs, loc=1)
    ax_ess.set_xlabel("Epoch")
    plt.close()

    display_id = display(fig, display_id=True)

    return dict(
        fig=fig,
        ax_ess=ax_ess,
        ax_loss=ax_loss,
        ess_line=ess_line,
        loss_line=loss_line,
        display_id=display_id,
        metric=metric,
    )


def moving_average(x, window=10):
    if len(x) < window:
        return np.mean(x, keepdims=True)
    else:
        return np.convolve(x, np.ones(window), "valid") / window


def update_plots(
    history, fig, ax_ess, ax_loss, ess_line, loss_line, metric, display_id
):
    Y = np.array(history["ess"])
    Y = moving_average(Y, window=15)
    ess_line[0].set_ydata(Y)
    ess_line[0].set_xdata(np.arange(len(Y)))
    Y = history[metric]
    Y = moving_average(Y, window=15)
    loss_line[0].set_ydata(np.array(Y))
    loss_line[0].set_xdata(np.arange(len(Y)))
    ax_loss.relim()
    ax_loss.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)
