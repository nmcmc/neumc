import re
import sys
from math import ceil

import torch

cuda_re = re.compile(r"cuda(:(\d+))?")


def cuda_requested(torch_device):
    return bool(cuda_re.match(torch_device))


def describe_device(device):
    if cuda_requested(device):
        props = torch.cuda.get_device_properties(device)
        return f"Running on {props.name} {ceil(props.total_memory / 2 ** 30)}GB GPU"
    else:
        if re.match("cpu", device):
            return "Running on CPU"
        else:
            return f"Unknown device {device}."


def check_cuda(device):
    if cuda_requested(device) and not torch.cuda.is_available():
        print("Requested cuda device but cuda is not available.")
        sys.exit(1)


def list_cuda_devices():
    if torch.cuda.is_available():
        for dev in range(torch.cuda.device_count()):
            print(f"cuda:{dev} ", torch.cuda.get_device_properties(dev))
    else:
        print("No cuda devices found.")
