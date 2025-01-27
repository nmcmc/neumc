from collections import OrderedDict

import torch


# Memory hooks
class PackHook:
    def __init__(self):
        self.tot_mem = 0

    def __call__(self, x):
        self.tot_mem += x.nelement() * x.element_size()
        return x

    def mem(self):
        return self.tot_mem


def unpack_hook(x):
    return x


class MemoryPackHook:
    def __init__(self):
        self.tot_mem = 0
        self.ptrs = []

    def __call__(self, x):
        size = x.nelement() * x.element_size()
        self.tot_mem += size
        self.ptrs.append((x.data_ptr(), size))
        return x

    def mem(self, *, b=0):
        unit = 2**b
        return self.tot_mem / unit

    def mem_u(self, *, b=0):
        unit = 2**b
        unique = set(self.ptrs)
        return sum([p[1] for p in unique]) / unit


SAVED_PREFIX = "_saved_"


def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX) :]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = "%-" + str(col1width) + "s: %" + str(col2width) + "s"
    truncate = lambda s: s[: col2width - 3] + "..." if len(s) > col2width else s
    params = "\n".join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + "\n" + sep + "\n" + params


class CallHook:
    def __init__(self):
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1


class TimingHook:
    def __init__(self):
        self.timer = torch.cuda.Event(enable_timing=True)

    def __call__(self, *args, **kwargs):
        self.timer.record()


class CountWalkHook:
    def __init__(self):
        self.count = 0

    def __call__(self, fn):
        self.count += 1


class GetNamesWalkHook:
    def __init__(self):
        self.fn_counts = {}

    def __call__(self, fn):
        name = get_fn_name(fn, False, 8)
        count = self.fn_counts.get(name, 0) + 1
        self.fn_counts[name] = count

    def names(self):
        return set(self.fn_counts)

    def n_nodes(self):
        n = 0
        for v in self.fn_counts.values():
            n += v
        return n


class RegisterHookWalkHook:
    def __init__(self, *, names=[], pre_hook=None, hook=None, create_new=False):
        self.names = names
        self.pre_hook = pre_hook
        self.hook = hook
        self.handles = []
        self.create_new = create_new
        self.hooks = OrderedDict()

    def __call__(self, fn):
        if self.names:
            name = get_fn_name(fn, False, 0)
            if name not in self.names:
                return
        if self.create_new:
            pre_hook = None
            hook = None
            if self.pre_hook is not None:
                pre_hook = self.pre_hook()
                self.handles.append(fn.register_prehook(pre_hook))
            if self.hook is not None:
                hook = self.hook()
                self.handles.append(fn.register_hook(hook))
            if not (hook is None and pre_hook is None):
                self.hooks[fn] = (pre_hook, hook)
        else:
            if self.pre_hook is not None:
                self.handles.append(fn.register_prehook(self.pre_hook))
            if self.hook is not None:
                self.handles.append(fn.register_hook(self.hook))

    def unregister(self):
        for h in self.handles:
            h.remove()


def walk_(fn, *, all=False, hook=None):
    seen = set()
    stack = []

    stack.append((fn, 0))
    depth = 0
    while stack:
        ff, d = stack.pop()
        depth = max(depth, d)
        if not all:
            if ff in seen:
                continue
            seen.add(ff)
        if hook is not None:
            hook(ff)

        if hasattr(ff, "next_functions"):
            for f in ff.next_functions:
                if f[0] is not None:
                    stack.append((f[0], d + 1))
    return depth, hook


def time(loss, *, names=[]):
    d, h = walk_(
        loss.grad_fn,
        hook=RegisterHookWalkHook(
            names=names, pre_hook=TimingHook, hook=TimingHook, create_new=True
        ),
    )
    loss.backward()

    timings = OrderedDict()
    h.unregister()

    tot = 0.0
    for k, v in h.hooks.items():
        t = v[0].timer.elapsed_time(v[1].timer)
        tot += t
        timings[k] = t

    return tot, timings


def collect_by_name(timings):
    cum_timings = OrderedDict()

    for k, t in timings.items():
        name = get_fn_name(k, False, 0)
        count, time = cum_timings.get(name, (0, 0.0))
        cum_timings[name] = (count + 1, time + t)

    return cum_timings


from operator import itemgetter


def order_by_time(timings):
    times = [(k, c, t) for k, (c, t) in timings.items()]
    return sorted(times, key=itemgetter(2), reverse=True)


def pprint(timings):
    for i, (k, c, t) in enumerate(timings):
        print(f"{i + 1:3d} {t:8.2f} {c:5d} {k:20s}")
