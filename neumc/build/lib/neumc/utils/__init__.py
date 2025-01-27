__all__ = [
    "batch_function",
    "checkpoint",
    "errors",
    "metrics",
    "profile",
    "cuda",
    "stats_utils",
    "grab",
    "format_time",
    "cpuinfo",
    "utils",
    "ess",
    "ess_lw",
    "dkl",
    "calc_fnc",
]

from . import batch_function
from . import checkpoint
from . import errors
from . import metrics
from . import profile
from . import cuda
from . import stats_utils
from . import cpuinfo
from . import utils
from . import calc_fnc
from .utils import ess, ess_lw, dkl


def grab(var):
    return var.detach().cpu().numpy()


def format_time(s):
    isec = int(s)
    secs = isec % 60
    mins = isec // 60
    hours = mins // 60
    mins %= 60

    return f"{hours:02d}:{mins:02d}:{secs:02d}"
