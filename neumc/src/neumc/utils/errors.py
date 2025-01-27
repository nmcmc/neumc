import torch

__DEBUG__ = True

dump_dir = None


def is_not(something, variables):
    for k, v in variables.items():
        if not torch.all(something(v)):
            return True
    return False


def is_not_finite(variables):
    return is_not(torch.isfinite, variables)


def is_model_not_finite(model, check_grad=False):
    for p in model.parameters():
        if not torch.all(torch.isfinite(p)):
            return True
        if check_grad:
            if not torch.all(torch.isfinite(p.grad)):
                return True
    return False


def are_grad_not_finite(model):
    for p in model.parameters():
        if not torch.all(torch.isfinite(p.grad)):
            return True
    return False


def dump_on_error(error_occured, variables, path, raise_assertion=True):
    if __DEBUG__:
        with torch.no_grad():
            if error_occured(variables):
                torch.save(variables, file_name(path))
                if raise_assertion:
                    raise AssertionError


def grads(model):
    return {i: p.grad.clone() for i, p in enumerate(model.parameters())}


def save_grads(model, path):
    torch.save(grads(model), path)


def file_name(name):
    if dump_dir is not None:
        return f"{dump_dir}/{name}"
    else:
        return name
