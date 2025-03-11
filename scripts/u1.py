#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training U(1) lattice gauge theory.

This script trains U1 pure gauge model.
After training, it generates samples and computes the free energy
and compares it to the exact value.
"""

import time
from pathlib import Path


from neumc.utils.utils import dkl, ess
from numpy import log
import torch

import neumc
import neumc.utils.metrics as metrics
import neumc.utils.stats_utils as stats_utils


print(f"Running on PyTorch {torch.__version__}")

if torch.cuda.is_available():
    torch_device = "cuda"
    print(f"Running on {torch.cuda.get_device_name()}")
else:
    torch_device = "cpu"

OUTPUT_DIR = "out_u1"
output_dir_path = Path(OUTPUT_DIR)
output_dir_path.mkdir(parents=True, exist_ok=True)

n_eras = 2
n_epochs_per_era = 100
print_freq = 25  # epochs

# Physical model parameters
L = 8
lattice_shape = (L, L)
beta = 1.0

# Training parameters

loss = "path_gradient"
lr = 0.001
batch_size = 2**10
n_batches = 1

# Final sampling parameters.
n_samples = 2**17
sampling_batch_size = 2**10
n_boot_samples = 100
boot_bin_size = 1

float_dtype = torch.float32

config = {
    "layers": {
        "n_layers": 16,
        "masking": "u1",
        "coupling": "cs",
        "nn": {"hidden_channels": [32, 32], "kernel_size": 3, "dilation": 1},
        "n_knots": 9,
        "float_dtype": "float32",
        "lattice_shape": [L, L],
    }
}

layers_cfg = config["layers"]
nn_cfg = layers_cfg["nn"]

action = neumc.physics.u1.U1GaugeAction(beta)


masks = neumc.nf.gauge_masks.u1_masks_gen(
    lattice_shape=lattice_shape, float_dtype=float_dtype, device=torch_device
)

# masks = neumc.nf.gauge_masks.schwinger_masks_with_2x1_loops(
#     plaq_mask_shape=lattice_shape,
#     link_mask_shape=(len(lattice_shape), *lattice_shape),
#     float_dtype=float_dtype,
#     device=torch_device,
# )

in_channels = 2
n_knots = layers_cfg["n_knots"]

# Prior
prior = neumc.nf.prior.MultivariateUniform(
    torch.zeros((2, *lattice_shape)),
    2 * torch.pi * torch.ones((2, *lattice_shape)),
    device=torch_device,
)


# Coupling layers
def make_plaq_coupling(mask):
    out_channels = 3 * (n_knots - 1) + 1
    net = neumc.nf.nn.make_conv_net(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=nn_cfg["hidden_channels"],
        kernel_size=nn_cfg["kernel_size"],
        use_final_tanh=False,
        dilation=nn_cfg["dilation"],
    )
    net.to(torch_device)

    return neumc.nf.cs_coupling.CSCoupling(net, mask, n_knots)


layers = neumc.nf.u1_equiv.make_u1_equiv_layers(
    loops_function=None,  # lambda x: [neumc.physics.u1.compute_u1_2x1_loops(x)],
    make_plaq_coupling=make_plaq_coupling,
    masks=masks,
    n_layers=config["layers"]["n_layers"],
    device=torch_device,
)

model = {"layers": layers, "prior": prior}

grad_estimator_name="RT"
grad_estimator = getattr(neumc.training.gradient_estimator, f"{grad_estimator_name}Estimator")(prior, layers, action)

history = {"dkl": [], "std_dkl": [], "loss": [], "ess": []}

optimizer = torch.optim.Adam(model["layers"].parameters(), lr=0.001)

elapsed_time = 0
start_time = time.time()

total_epochs = n_eras * n_epochs_per_era
epochs_done = 0
print(f"Starting training: {n_eras} x {n_epochs_per_era} epochs")
for era in range(n_eras):
    for epoch in range(n_epochs_per_era):
        optimizer.zero_grad()
        loss_, logq, logp = grad_estimator.step(batch_size)
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
                path=f"{OUTPUT_DIR}/u1_{grad_estimator_name}_{L:02d}x{L:02d}.zip"
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

# Sampling and free energy estimation

if n_samples > 0:
    print(f"Sampling {n_samples} configurations")
    if neumc.physics.u1.scipy_installed:
        F_exact = -neumc.physics.u1.logZ(L, beta=beta) - 2 * L * L * log(2 * torch.pi)
    else:
        F_exact = None
    u, lq = neumc.nf.flow.sample(
        batch_size=sampling_batch_size, n_samples=n_samples, prior=prior, layers=layers
    )
    lp = -action(u)
    lw = lp - lq
    F_q, F_q_std = stats_utils.torch_bootstrapf(
        lambda x: -torch.mean(x), lw, n_samples=n_boot_samples, binsize=boot_bin_size
    )
    F_nis, F_nis_std = stats_utils.torch_bootstrapf(
        lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))),
        lw,
        n_samples=n_boot_samples,
        binsize=boot_bin_size,
    )

    if F_exact is not None:
        print(
            f"Free energy true = {F_exact:.4f} variational = {F_q:.4f}+/-{F_q_std:.4f} diff = {F_q - F_exact:.4f}"
        )
        print(
            f"Free energy true = {F_exact:.4f} NIS         = {F_nis:.4f}+/-{F_nis_std:.4f} diff = {F_nis - F_exact:.4f}"
        )
        if torch.abs(F_nis - F_exact) > 4 * F_nis_std:
            print("NIS free energy is not consistent with the exact value.")
    else:
        print(f"Free energy variational = {F_q:.4f}+/-{F_q_std:.4f}")
        print(f"Free energy NIS         = {F_nis:.4f}+/-{F_nis_std:.4f} ")
