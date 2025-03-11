#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import torch
from numpy import log

import neumc
import neumc.nf.cs_coupling as csc
import neumc.physics.u1 as u1
from neumc.nf.u1_equiv import U1GaugeEquivCouplingLayer

parser = argparse.ArgumentParser("Schwinger")

parser.add_argument("--batch-size", type=int, default=2 ** 10, help="Batch size for training")
parser.add_argument("--n-batches", type=int, default=1)
parser.add_argument("--n-eras", type=int, default=4, help="Number of training eras")
parser.add_argument("--n-updates-per-era", type=int, default=50, help="Number of batch updates per era")

args = parser.parse_args()

if torch.cuda.is_available():
    torch_device = "cuda"
    print(f"Running on {torch.cuda.get_device_name()}")
else:
    torch_device = "cpu"
    print(f"Running on {neumc.utils.cpuinfo.get_processor_name()} CPU")
    print("Warning running on CPU will be much to slow")

float_dtype = torch.float32

OUTPUT_DIR = "out_schwinger"
output_dir_path = Path(OUTPUT_DIR)
output_dir_path.mkdir(parents=True, exist_ok=True)

L = 8
lattice_shape = (L, L)

# Action
beta = 1.0
kappa = 0.276
qed_action = neumc.physics.schwinger.QEDAction(beta=beta, kappa=kappa)

# Masks

masks = neumc.nf.gauge_masks.sch_2x1_masks_gen(
    lattice_shape=lattice_shape,
    float_dtype=float_dtype,
    device=torch_device,
)
loops_function = lambda x: [u1.compute_u1_2x1_loops(x)]
in_channels = 6

# Circular splines
n_knots = 9
out_channels = 3 * (n_knots - 1) + 1

# Model
n_layers = 48

# Neural network

hidden_channels = [64, 64]
kernel_size = 3
dilation = [1, 2, 3]

model_cfg = {
    "n_layers": n_layers,
    "lattice_shape": lattice_shape,
    "hidden_channels": hidden_channels,
    "kernel_size": kernel_size,
    "dilation": dilation
}

layers = []
for l in range(n_layers):
    net = neumc.nf.nn.make_conv_net(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        use_final_tanh=False,
        dilation=dilation)
    net.to(torch_device)

    link_mask, plaq_mask = next(masks)
    conditioner = csc.CSConditioner(net, n_knots=n_knots)
    plaq_coupling = csc.CouplingLayer(
        conditioner=conditioner,
        transform=csc.CSTransform(),
        mask=plaq_mask
    )
    link_coupling = U1GaugeEquivCouplingLayer(
        loops_function=loops_function,
        active_links_mask=link_mask,
        plaq_coupling=plaq_coupling)
    layers.append(link_coupling)

model = neumc.nf.flow_abc.TransformationSequence(layers)

prior = neumc.nf.prior.MultivariateUniform(
    torch.zeros((2, *lattice_shape), device=torch_device),
    2 * torch.pi *
    torch.ones(lattice_shape, device=torch_device)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

batch_size = args.batch_size
n_batches = args.n_batches

grad_estimator_name = "REINFORCE"
grad_estimator = getattr(neumc.training.gradient_estimator, f"{grad_estimator_name}Estimator")(
    prior=prior,
    flow=model,
    action=qed_action,
    use_amp=False)

n_eras = args.n_eras
n_epochs_per_era = args.n_updates_per_era

print(f"Starting training: {n_eras} x {n_epochs_per_era} epochs")
print_frequency = args.n_updates_per_era // 4
elapsed_time = 0
start_time = time.time()
for era in range(n_eras):
    total_ess = 0.0
    print(f"Starting era {era}")
    for epoch in range(n_epochs_per_era):
        optimizer.zero_grad()
        loss_, logq, logp = grad_estimator.step(
            batch_size=batch_size,
            n_batches=n_batches)
        total_ess += neumc.utils.ess(logp, logq)
        optimizer.step()
        if (epoch + 1) % print_frequency == 0:
            print(f"  Finished epoch {epoch}")
            neumc.utils.checkpoint.safe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                era=era,
                model_cfg=model_cfg,
                **{"beta": beta, "kappa": kappa},
                path=f"{OUTPUT_DIR}/sch_{grad_estimator_name}_{L:02d}x{L:02d}.zip",
            )

    total_ess /= n_epochs_per_era
    elapsed_time = time.time() - start_time
    print(f"Finished era {era} ESS = {total_ess.item():.4f} elapsed time {elapsed_time:.1f}s")

n_samples = 2 ** 12
n_boot_samples = 100

if n_samples > 0:
    bin_size = 1
    batch_size = 512
    print(f"Sampling {n_samples} configurations")

    u, lq = neumc.nf.flow.sample(
        batch_size=batch_size, n_samples=n_samples, prior=prior, layers=model
    )
    lp = -qed_action(u)
    lw = lp - lq
    F_q, F_q_std = neumc.utils.stats_utils.torch_bootstrapf(
        lambda x: -torch.mean(x), lw, n_samples=n_boot_samples, binsize=bin_size
    )
    F_nis, F_nis_std = neumc.utils.stats_utils.torch_bootstrapf(
        lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))),
        lw,
        n_samples=n_boot_samples,
        binsize=bin_size,
    )

    print(f"Free energy variational = {F_q:.4f}+/-{F_q_std:.4f}")
    print(f"Free energy NIS         = {F_nis:.4f}+/-{F_nis_std:.4f}")
