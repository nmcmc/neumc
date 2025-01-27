#
# Author: Piotr Białas
#
"""

This module provides utility functions for loading configurations from yaml files and command line arguments using the
omegaconf library. The main function is `get_config` that loads the configuration from the command line arguments and
yaml files.
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from collections.abc import Mapping, Sequence


def _add_suffix(path: Path, suffix) -> Path:
    """
    Adds suffix to the path if it does not have one.

    Parameters
    ----------
    path
    suffix

    Returns
    -------
    Path with suffix added
    """

    if not path.suffix:
        path = path.with_suffix(suffix)
    return path


def load_configs(paths: Sequence[str], *, verbose=0) -> DictConfig:
    r"""
    Load configurations from yaml files and merge them.

    The configurations are merged in the order they are provided in the list.

    Parameters
    ----------
    paths: list of str
        list of paths to yaml files. The \*.yaml suffix may be omitted.
    verbose

    Returns
    -------
    DictConfig  object with the merged configurations
    """
    conf = OmegaConf.create()
    for p in paths:
        path = Path(p)
        path = _add_suffix(path, ".yaml")
        if verbose > 0:
            print(f"Loading configuration from {path}")
        try:
            loaded_conf = OmegaConf.load(path)
        except Exception as e:
            print(f"Error loading configuration from {path}")
            print(e)
            raise (e)
        conf.merge_with(loaded_conf)
    return conf


def get_config(*default_configs: str, verbose=0) -> DictConfig:
    r"""
    Load configuration from the command line arguments and yaml files.

    First the configuration from  CLI arguments is loaded. If it contains a 'config' key, a configuration is loaded
    from corresponding yaml files. If not the configuration is loaded from the default_configs yaml files. Then the
    cli configuration is merged with the configurations provided in the files. In that way the configuration given on
    the CL will supersede the configuration from the files.

    Parameters
    ----------
    default_configs
        paths to yaml files. The \*.yaml suffix may be omitted.
    verbose: int
        verbosity level

    Returns
    -------
    DictConfig object with the merged configurations
    """

    cli_conf = OmegaConf.from_cli()

    if "config" in cli_conf:
        paths = cli_conf.config
        if not isinstance(paths, Sequence):
            paths = (paths,)
    else:
        paths = default_configs

    try:
        if verbose > 0:
            print(f"Loading configuration from {paths}")
        conf = load_configs(paths, verbose=verbose)
    except Exception as e:
        print(f"Error loading configuration from {paths}")
        print(e)
        raise (e)

    conf.merge_with(cli_conf)
    return conf


# write code that finds a key in nested dictionary
def find_key(d, key):
    doted_key = ""
    for k, v in d.items():
        if k == key:
            doted_key = key + "." + doted_key if doted_key else key
            return doted_key, v
        if isinstance(v, Mapping):
            item = find_key(v, key)
            if item is not None:
                return item
    return None


# write code that finds a doted list key in dictionary
def find_doted_key(d, key):
    keys = key.split(".")
    for k, v in d.items():
        if k == keys[0]:
            if len(keys) == 1:
                return v
            else:
                return find_key(v, ".".join(keys[1:]))
        if isinstance(v, Mapping):
            item = find_key(v, key)
            if item is not None:
                return item
    return None
