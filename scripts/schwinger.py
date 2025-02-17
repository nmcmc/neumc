#!/usr/bin/env python3
"""Training a Schwinger model

This script trains a Schwinger model with two flavours of Wilson fermions based on the paper
https://arxiv.org/abs/2202.11712

If n-samples option is set to a non-zero value, the script will generate samples and compute the free energy.

It periodically saves the model and optimizer state to a zip file. The model can then be resumed from the checkpoint
using the --continue option.
"""

import time
import argparse

from neumc.training.gradient_estimator import REINFORCEEstimator
from neumc.utils.utils import dkl, ess
from numpy import log
import torch

import neumc
import neumc.utils.metrics as metrics
from neumc.nf.u1_model_asm import assemble_model_from_dict
import neumc.physics.schwinger as sch
import neumc.utils.stats_utils as stats_utils

parser = argparse.ArgumentParser("Schwinger")

parser.add_argument("--batch-size", type=int, default=2**10, help="Batch size for training")
parser.add_argument("--n-batches", type=int, default=1)
parser.add_argument("--n-eras", type=int, default=4, help="Number of training eras")
parser.add_argument("-n-updates-per-era", type=int, default=50, help="Number of batch updates per era")

args = parser.parse_args()


if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"
    print("Warning running on CPU will be much to slow")

L = 8

float_dtype = torch.float32


config = {
    "device": torch_device,
    "lattice_size": L,
    "verbose": 1,
    "phys_model": {"name": "U1", "parameters": {"beta": 1.0, "kappa": 0.276}},
    "model": {
        "n_layers": 48,
        "masking": "schwinger",
        "coupling": "cs",
        "nn": {
            "hidden_channels": [64, 64],
            "kernel_size": 3,
            "dilation": [1, 2, 3],
        },
        "n_knots": 9,
        "float_dtype": torch.float32,
        "lattice_shape": [L, L],
    },
    "train": {
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "loss": "REINFORCE",
        "n_eras": args.n_eras,
        "n_epochs_per_era": args.n_updates_per_era,
        "optimizer": {"name": "Adam", "kwargs": {"lr": 0.00025}},
    },
    "sampling": {
        "n_samples": 2**12,
        "batch_size": 512,
        "bootstrap": {"n_samples": 100, "bin_size": 1},
    },
    "io": {
        "output_dir": "out",
    },
}

print_freq = 25

if __name__ == "__main__":
    print(
        f"beta = {config['phys_model']['parameters']['beta']} kappa = {config['phys_model']['parameters']['kappa']}"
    )
    model = assemble_model_from_dict(config["model"], device=torch_device, verbose=1)
    layers = model["layers"]
    prior = model["prior"]
    action = sch.QEDAction(**config["phys_model"]["parameters"])

    gradient_estimator = REINFORCEEstimator(prior, layers, action, False)
    optimizer = getattr(torch.optim, config["train"]["optimizer"]["name"])(
        model["layers"].parameters(), **config["train"]["optimizer"]["kwargs"]
    )

    history = {"dkl": [], "std_dkl": [], "loss": [], "ess": []}

    train_cfg = config["train"]
    batch_size = train_cfg["batch_size"]
    elapsed_time = 0
    start_time = time.time()

    n_eras = train_cfg["n_eras"]
    n_epochs_per_era = train_cfg["n_epochs_per_era"]
    total_epochs = n_eras * n_epochs_per_era
    epochs_done = 0
    output_file = (
        f"{config['io']['output_dir']}u1_{train_cfg['loss']}_{L:02d}x{L:02d}.zip"
    )
    print(f"Starting training: {n_eras} x {n_epochs_per_era} epochs")
    for era in range(n_eras):
        for epoch in range(n_epochs_per_era):
            optimizer.zero_grad()
            loss_, logq, logp = gradient_estimator.step(batch_size)
            optimizer.step()
            metrics.add_metrics(
                history,
                {
                    "loss": loss_.cpu().numpy().item(),
                    "ess": ess(logp, logq).cpu().numpy().item(),
                    "dkl": dkl(logp, logq).cpu().numpy().item(),
                    "std_dkl": (logp - logq).std().cpu().numpy().item(),
                },
            )
            epochs_done += 1
            if (epoch + 1) % print_freq == 0:
                neumc.utils.checkpoint.safe_save_checkpoint(
                    model=layers,
                    optimizer=optimizer,
                    scheduler=None,
                    era=era,
                    configuration=config,
                    path=output_file,
                )
                elapsed_time = time.time() - start_time
                avg = metrics.average_metrics(history, n_epochs_per_era, history.keys())

                print(
                    f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}",
                    end="",
                )
                if epochs_done > 0:
                    time_per_epoch = elapsed_time / epochs_done
                    time_remaining = (total_epochs - epochs_done) * time_per_epoch

                    print(
                        f" {time_per_epoch:.2f}s/epoch  remaining time {neumc.utils.format_time(time_remaining):s}"
                    )
                    metrics.print_dict(avg)

    print(f"{elapsed_time / n_eras:.2f}s/era")

    if (n_samples := config["sampling"]["n_samples"]) > 0:
        n_boot_samples = config["sampling"]["bootstrap"]["n_samples"]
        bin_size = config["sampling"]["bootstrap"]["bin_size"]
        batch_size = config["sampling"]["batch_size"]
        print(f"Sampling {n_samples} configurations")

        u, lq = neumc.nf.flow.sample(
            batch_size=batch_size, n_samples=n_samples, prior=prior, layers=layers
        )
        lp = -action(u)
        lw = lp - lq
        F_q, F_q_std = stats_utils.torch_bootstrapf(
            lambda x: -torch.mean(x), lw, n_samples=n_boot_samples, binsize=bin_size
        )
        F_nis, F_nis_std = stats_utils.torch_bootstrapf(
            lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))),
            lw,
            n_samples=n_boot_samples,
            binsize=bin_size,
        )

        print(f"Free energy variational = {F_q:.4f}+/-{F_q_std:.4f}")
        print(f"Free energy NIS         = {F_nis:.4f}+/-{F_nis_std:.4f}")
