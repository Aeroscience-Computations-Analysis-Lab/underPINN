"""YAML config loader for underPINN.

Public API
----------
load_config(path)              → SimpleNamespace  (single experiment)
generate_sweep_configs(path)   → list[SimpleNamespace]  (Cartesian sweep)
cfg_get(ns, *attrs, default)   → value  (safe nested access)
save_config(ns, path)          → None  (write resolved config to YAML)
merge_config(ns, overrides)    → SimpleNamespace  (apply dot-key overrides)
"""

import copy
import types
from itertools import product

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "pyyaml is required for YAML config support. "
        "Install it with:  pip install pyyaml"
    ) from e


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_ns(data):
    """Recursively convert a YAML-loaded value to nested SimpleNamespace.

    YAML scalars used as mapping keys (e.g. ``2000: 0.5``) are parsed by
    PyYAML as Python ints, not strings.  ``SimpleNamespace`` requires string
    keyword arguments, so we stringify all keys here.  Any code that reads
    such a sub-namespace should access the field as ``ns.__dict__`` or use
    ``cfg_get`` rather than attribute access when keys were originally integers.
    """
    if isinstance(data, dict):
        return types.SimpleNamespace(**{str(k): _to_ns(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_to_ns(v) for v in data]
    return data


def _ns_to_dict(ns):
    """Recursively convert a SimpleNamespace back to a plain dict."""
    if isinstance(ns, types.SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_ns_to_dict(v) for v in ns]
    return ns


def _set_nested_dict(d: dict, dotted_key: str, value) -> None:
    """Write *value* into nested dict *d* using a dot-separated key path.

    Example::

        _set_nested_dict(d, "training.lr", 3e-4)
        # equivalent to d["training"]["lr"] = 3e-4
    """
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(path: str) -> types.SimpleNamespace:
    """Load a YAML experiment config and return a nested SimpleNamespace.

    Every nested mapping becomes a ``SimpleNamespace``; lists and scalars
    are left as-is.  Missing keys raise ``AttributeError`` on access —
    use :func:`cfg_get` for optional fields.

    Parameters
    ----------
    path : path to the ``.yaml`` / ``.yml`` file

    Returns
    -------
    types.SimpleNamespace
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data)}")
    return _to_ns(data)


def cfg_get(ns, *attrs, default=None, warn: bool = False):
    """Safely traverse a nested SimpleNamespace chain.

    Returns *default* if any level of the path is missing.

    Parameters
    ----------
    ns      : root SimpleNamespace (usually the top-level config)
    *attrs  : sequence of attribute names forming the key path
    default : value returned when the path is absent
    warn    : if ``True``, emit a :mod:`warnings` warning whenever the
              default is used because an attribute was not found.  Useful
              for catching typos in field names (e.g. ``"n_colocation"``
              instead of ``"n_collocation"``).

    Example::

        out_dir = cfg_get(cfg, "output", "dir", default="outputs/run")
        patience = cfg_get(cfg, "training", "early_stopping_patience", default=None)
        n_col    = cfg_get(cfg.data, "n_collocation", default=5000, warn=True)
    """
    import warnings as _warnings
    obj = ns
    for i, a in enumerate(attrs):
        try:
            obj = getattr(obj, a)
        except AttributeError:
            if warn:
                path = ".".join(str(x) for x in attrs[: i + 1])
                _warnings.warn(
                    f"cfg_get: '{path}' not found in config — "
                    f"using default={default!r}",
                    stacklevel=2,
                )
            return default
    return default if obj is None else obj


def save_config(ns: types.SimpleNamespace, path: str) -> None:
    """Serialise a SimpleNamespace config back to a YAML file.

    The output directory is created if it does not exist.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(_ns_to_dict(ns), fh, default_flow_style=False, sort_keys=False)


def merge_config(base_ns: types.SimpleNamespace,
                 overrides: dict) -> types.SimpleNamespace:
    """Return a new config with *overrides* applied.

    Keys in *overrides* may use dot notation to target nested fields::

        new_cfg = merge_config(cfg, {"physics.nu": 0.05, "training.lr": 3e-4})
    """
    d = copy.deepcopy(_ns_to_dict(base_ns))
    for k, v in overrides.items():
        _set_nested_dict(d, k, v)
    return _to_ns(d)


def generate_sweep_configs(path: str) -> list:
    """Expand a sweep YAML into one config per hyperparameter combination.

    Sweep YAML format
    -----------------
    ::

        base:                        # full base config (required)
          problem: burgers
          physics:
            nu: 0.01
          training:
            epochs: 5000
            lr: 1.0e-3
          ...

        sweep:                       # dot-separated paths → list of values
          physics.nu: [0.01, 0.025, 0.05]
          training.lr: [1.0e-3, 3.0e-4]

    The returned list contains one ``SimpleNamespace`` per row of the
    Cartesian product, with each run's parameters applied on top of
    ``base``.  If ``sweep`` is absent the list contains a single config
    equal to ``base``.

    Parameters
    ----------
    path : path to the sweep ``.yaml`` file
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    base_dict  = data.get("base",  {})
    sweep_dict = data.get("sweep", {})

    if not sweep_dict:
        return [_to_ns(base_dict)]

    keys   = list(sweep_dict.keys())
    values = [v if isinstance(v, list) else [v] for v in sweep_dict.values()]

    configs = []
    for combo in product(*values):
        cfg_dict = copy.deepcopy(base_dict)
        for k, v in zip(keys, combo):
            _set_nested_dict(cfg_dict, k, v)
        configs.append(_to_ns(cfg_dict))

    return configs
