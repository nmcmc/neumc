from warnings import warn


def calc_function(cfgs, *, function, batch_size, device, **kwargs):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.batch_function import batch_function

    return batch_function(
        cfgs, function=function, batch_size=batch_size, device=device, **kwargs
    )


def calc_action(cfgs, *, action, batch_size, device):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.batch_function import batch_action

    return batch_action(cfgs, action=action, batch_size=batch_size, device=device)


def compute_ess_lw(logw):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.utils import ess_lw

    return ess_lw(logw)


def compute_ess_lw_np(logw):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.utils import ess_lw

    return ess_lw(logw)


def compute_ess(logp, logq):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.utils import ess

    return ess(logp, logq)


def calc_dkl(logp, logq):
    warn(
        "This file is depreciated. Functionality moved to neumc.utils or neumc.utils.batch_function.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neumc.utils.utils import dkl

    return dkl(logp, logq)
