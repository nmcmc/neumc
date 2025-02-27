#!/usr/bin/env python3

import argparse

import torch

import neumc
import neumc.nf.cs_coupling as csc
import neumc.physics.u1 as u1
from neumc.nf.u1_equiv import U1GaugeEquivCouplingLayer
from neumc.training.gradient_estimator import REINFORCEEstimator

parser = argparse.ArgumentParser("Schwinger")

parser.add_argument("--batch-size", type=int, default=2 ** 10, help="Batch size for training")
parser.add_argument("--n-batches", type=int, default=1)
parser.add_argument("--n-eras", type=int, default=4, help="Number of training eras")
parser.add_argument("-n-updates-per-era", type=int, default=50, help="Number of batch updates per era")

args = parser.parse_args()

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"
    print("Warning running on CPU will be much to slow")

float_dtype = torch.float32

L = 8
lattice_shape = (L, L)

# Action

qed_action = neumc.physics.schwinger.QEDAction(beta=1.0, kappa=0.276)

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

#
# Model

n_layers = 48

# Neural network

hidden_channels = [64, 64]
kernel_size = 3
dilation = [1, 2, 3]

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

grad_estimator = REINFORCEEstimator(
    prior=prior,
    flow=model,
    action=qed_action,
    use_amp=False)

n_eras = args.n_eras
n_epochs_per_era = args.n_updates_per_era

for era in range(n_eras):
    total_ess = 0.0
    for epoch in range(n_epochs_per_era):
        optimizer.zero_grad()
        loss_, logq, logp = grad_estimator.step(
            batch_size=batch_size,
            n_batches=n_batches)
        total_ess += neumc.utils.ess(logp, logq)
        optimizer.step()
    total_ess /= n_epochs_per_era
    print(era, total_ess)
