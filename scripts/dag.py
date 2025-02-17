#!/usr/bin/env python3
"""Analyzing DAG for Schwinger model

This script analyzes the computational graph for the Schwinger model. It calculates the number of nodes,
as well as the height and width of the graph.

It also  counts the number of nodes of given type (operation) and measures the time needed to complete each operation.
"""

import argparse
import sys
import re

import neumc.training.gradient_estimator
import torch


from neumc.physics import schwinger as sch
from neumc.nf.u1_model_asm import assemble_model_from_dict

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to run the training on."
)
parser.add_argument(
    "-L",
    "--lattice-size",
    type=int,
    action="store",
    default=8,
    help="L - lattice size will be LxL",
)
parser.add_argument(
    "--batch-size", type=int, action="store", default="256", help="Size of the batch"
)
parser.add_argument(
    "--loss", default="REINFORCE", choices=["REINFORCE", "Rt"], help="Loss function"
)
parser.add_argument(
    "--float-dtype",
    default="float32",
    choices=["float32", "float64"],
    help="Float precision used for training",
)
parser.add_argument("-b", "--beta", type=float, default="2.0", help="beta")
parser.add_argument("-k", "--kappa", type=float, default="0.276", help="kappa")
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default="1",
    help="Verbosity level if zero does not print anything.",
)

args = parser.parse_args()

if args.verbose > 0:
    print(f"Running on PyTorch {torch.__version__}")

torch_device = args.device

if re.match(r"cuda", torch_device):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. Exiting.")
        sys.exit(1)

if args.verbose > 0:
    print(torch.cuda.get_device_name())

batch_size = args.batch_size
float_dtype = args.float_dtype


L = args.lattice_size
lattice_shape = (L, L)
link_shape = (2, L, L)
beta = args.beta
kappa = args.kappa
qed_action = sch.QEDAction(beta, kappa)

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

model = assemble_model_from_dict(model_cfg, torch_device, verbose=args.verbose)

layers = model["layers"]
prior = model["prior"]

gradient_estimator = getattr(
    neumc.training.gradient_estimator, f"{args.loss}Estimator"
)(prior, layers, qed_action, False)
z = prior.sample_n(8)
log_prob_z = prior.log_prob(z)

l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)
d, h = neumc.utils.profile.walk_(l.grad_fn, hook=neumc.utils.profile.CountWalkHook())
l = None
print(f"Total number of nodes in DAG is {h.count} height = {d}")

l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)
d, h = neumc.utils.profile.walk_(l.grad_fn, hook=neumc.utils.profile.GetNamesWalkHook())
l = None
print(f"Total number of different operations in DAG is {d}")
print(h.fn_counts)

l, logq, logp = gradient_estimator.forward_pass(z, log_prob_z)
tot, timings = neumc.utils.profile.time(l, names=[])
l = None
print(f"Total time taken by all operation is {tot}ms")
neumc.utils.profile.pprint(
    neumc.utils.profile.order_by_time(neumc.utils.profile.collect_by_name(timings))
)
