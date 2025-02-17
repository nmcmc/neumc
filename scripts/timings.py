#!/usr/bin/env python3
"""Detailed timing measurements for loss functions.

This script measures the time of the loss function evaluation and the time of the backward pass.
It first performs  several warmup steps and then measures the time of the loss function evaluation and the backward pass
for a given number of measurements reporting the mean and the standard deviation of the measurements.

The model parameters can be specified in a JSON file. The default parameters are used if no file is specified.
"""

from neumc.training.gradient_estimator import REINFORCEEstimator
import torch
import numpy as np


import neumc
from neumc.nf.u1_model_asm import assemble_model_from_dict

verbose = 1

warming = 5
measurements = 40
include_prior_generation = True

batch_size = 2**8

L = 8

lattice_shape = (L, L)
beta = 1.0
kappa = 0.276

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"

qed_action = neumc.physics.schwinger.QEDAction(beta, kappa)

model_cfg = {
    "n_layers": 48,
    "masking": "2x1",
    "coupling": "cs",
    "nn": {
        "hidden_channels": [64, 64],
        "kernel_size": 3,
        "dilation": [1, 2, 3],
    },
    "n_knots": 9,
    "float_dtype": torch.float32,
    "lattice_shape": lattice_shape,
}

model = assemble_model_from_dict(model_cfg, device=torch_device, verbose=verbose)

layers = model["layers"]

prior = model["prior"]

gradient_estimator = REINFORCEEstimator(prior, layers, qed_action, False)
z = prior.sample_n(batch_size)
log_prob_z = prior.log_prob(z)

starter, ender = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
loss_starter, loss_ender = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
back_starter, back_ender = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
timings = np.zeros((measurements, 3))
if verbose > 0:
    print("warming ... ")
for i in range(warming):
    if verbose > 0:
        if i % 5 == 0:
            print(i + 1)
    if include_prior_generation:
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
    l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)
    l.backward()
if verbose > 0:
    print("measuring ... ")

# WAIT FOR GPU SYNC
torch.cuda.synchronize()
for r in range(measurements):
    if verbose > 0:
        if r % 5 == 0:
            print(r + 1)
    starter.record()
    loss_starter.record()
    if include_prior_generation:
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
    l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)
    loss_ender.record()
    back_starter.record()
    l.backward()
    back_ender.record()
    ender.record()

    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    timings[r, 0] = starter.elapsed_time(ender)
    timings[r, 1] = loss_starter.elapsed_time(loss_ender)
    timings[r, 2] = back_starter.elapsed_time(back_ender)


def parameters_string():
    return f"L {L} batch-size {batch_size} loss {gradient_estimator.__class__.__name__}"


print(
    parameters_string(),
    " tot. {0:.2f}+/-{3:.3f} loss  {1:.2f}+/-{4:.3f} back. {2:.2f}+/-{5:.3f}".format(
        *(timings.mean(0) / 1000), *(timings.std(0) / 1000)
    ),
    flush=True,
)
