#!/usr/bin/env python3
"""Memory usage analysis for Schwinger model

This script checks amount  memory allocated while evaluating a loss function for Schwinger model.
"""

import torch

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"

import neumc

print(f"Running on PyTorch {torch.__version__}")

batch_size = 2 ** 8
float_dtype = "float32"

L = 8
lattice_shape = (L, L)
link_shape = (2, L, L)
beta = 1.0
kappa = 0.276

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

model = neumc.nf.u1_model_asm.assemble_model_from_dict(
    model_cfg, torch_device, verbose=1
)
layers = model["layers"]
prior = model["prior"]

gradient_estimator = neumc.training.gradient_estimator.REINFORCEEstimator(
    prior, layers, qed_action, False)

print(f"L = {L} Estimator {gradient_estimator.__class__.__name__}")

z = prior.sample_n(batch_size=batch_size)
log_prob_z = prior.log_prob(z)

if torch_device == "cuda":
    torch.cuda.reset_peak_memory_stats()

with torch.autograd.graph.saved_tensors_hooks(
        pack_hook := neumc.utils.profile.MemoryPackHook(),
        neumc.utils.profile.unpack_hook):
    l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)

print(
    f"{pack_hook.mem_u(b=30):.2f}GB memory used for storing {len(set(pack_hook.ptrs))} tensores in the DAG"
)

if torch_device == "cuda":
    print(f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA")

l.backward()

print(
    f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA after running backward()"
)
