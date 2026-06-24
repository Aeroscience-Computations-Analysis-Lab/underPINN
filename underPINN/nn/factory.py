"""Central network factory — build a Flax model from a plain config dict.

A single mapping from ``network`` config → model, used by both training runners
and the prediction utilities so a new architecture round-trips automatically
(the saved ``network`` metadata is exactly the dict this consumes).

``build_model`` takes a plain dict (e.g. the ``network`` block saved in a
checkpoint's metadata).  ``network_config`` extracts that dict from a loaded
config object, filling architecture-specific fields — notably deriving the
temporal base frequency ω = 2π/T_period for the temporal-Fourier network.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from underPINN.nn.mlp import MLP, GatedMLP, FourierMLP, TemporalFourierMLP, SIREN

_TEMPORAL = ("temporal_fourier", "fourier_time", "tfmlp")


def build_model(net_cfg: Dict[str, Any]):
    """Build a Flax module from a ``network`` config dict.

    Recognised ``type`` values: ``mlp``, ``gated_mlp``, ``fourier_mlp``,
    ``temporal_fourier``.  Unknown types fall back to a plain MLP.
    """
    t      = str(net_cfg.get("type", "mlp")).lower()
    layers = list(net_cfg["layers"])

    if t in _TEMPORAL:
        return TemporalFourierMLP(
            layers=layers,
            omega=float(net_cfg.get("omega", 2.0 * math.pi)),
            n_time_harmonics=int(net_cfg.get("n_time_harmonics", 4)),
            time_index=int(net_cfg.get("time_index", -1)),
            gated=bool(net_cfg.get("gated", True)),
        )
    if t == "gated_mlp":
        return GatedMLP(layers=layers)
    if t == "siren":
        return SIREN(layers=layers, w0=float(net_cfg.get("w0", 15.0)))
    if t in ("fourier_mlp", "fourier"):
        return FourierMLP(layers=layers,
                          n_fourier=int(net_cfg.get("n_fourier", 16)),
                          sigma=float(net_cfg.get("sigma", 1.0)))
    return MLP(layers=layers)


def network_config(cfg) -> Dict[str, Any]:
    """Extract the ``network`` config dict from a loaded config object.

    For the temporal-Fourier network the base frequency ω is taken from
    ``network.omega`` if given, else derived as 2π/``physics.T_period``.  The
    returned dict is what should be both fed to :func:`build_model` and saved as
    the run's ``network`` metadata (so prediction rebuilds the same model).
    """
    from underPINN.config.loader import cfg_get

    nc = cfg.network
    out: Dict[str, Any] = {
        "type":   str(cfg_get(nc, "type", default="gated_mlp")).lower(),
        "layers": list(nc.layers),
    }
    if out["type"] in _TEMPORAL:
        out["n_time_harmonics"] = int(cfg_get(nc, "n_time_harmonics", default=4))
        out["time_index"]       = int(cfg_get(nc, "time_index", default=-1))
        out["gated"]            = bool(cfg_get(nc, "gated", default=True))
        omega = cfg_get(nc, "omega", default=None)
        if omega is None:
            ph = cfg_get(cfg, "physics", default=None)
            T_period = float(cfg_get(ph, "T_period", default=1.0)) if ph is not None else 1.0
            omega = 2.0 * math.pi / T_period
        out["omega"] = float(omega)
    elif out["type"] == "siren":
        out["w0"] = float(cfg_get(nc, "w0", default=15.0))
    elif out["type"] in ("fourier_mlp", "fourier"):
        out["n_fourier"] = int(cfg_get(nc, "n_fourier", default=16))
        out["sigma"]     = float(cfg_get(nc, "sigma", default=1.0))
    return out
