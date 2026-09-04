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
_OPERATORS = ("fno1d", "fno2d", "deeponet", "cvit")


def build_model(net_cfg: Dict[str, Any]):
    """Build a Flax module from a ``network`` config dict.

    Recognised ``type`` values: ``mlp``, ``gated_mlp``, ``fourier_mlp``,
    ``temporal_fourier``, ``siren``.  Unknown types fall back to a plain MLP.

    Point networks read ``layers`` (``[in, h1, …, out]``); architectures that
    do not use a flat layer list (e.g. future neural operators) read their own
    keys, so ``layers`` is only required by the branches that consume it.
    """
    t = str(net_cfg.get("type", "mlp")).lower()

    def _layers():
        if "layers" not in net_cfg:
            raise KeyError(f"network.type='{t}' requires a 'layers' list")
        return list(net_cfg["layers"])

    if t in _TEMPORAL:
        return TemporalFourierMLP(
            layers=_layers(),
            omega=float(net_cfg.get("omega", 2.0 * math.pi)),
            n_time_harmonics=int(net_cfg.get("n_time_harmonics", 4)),
            time_index=int(net_cfg.get("time_index", -1)),
            gated=bool(net_cfg.get("gated", True)),
        )
    if t == "gated_mlp":
        return GatedMLP(layers=_layers())
    if t == "siren":
        return SIREN(layers=_layers(), w0=float(net_cfg.get("w0", 15.0)))
    if t in ("fourier_mlp", "fourier"):
        return FourierMLP(layers=_layers(),
                          n_fourier=int(net_cfg.get("n_fourier", 16)),
                          sigma=float(net_cfg.get("sigma", 1.0)))
    if t == "fno1d":
        from underPINN.nn.operators import FNO1D
        return FNO1D(
            modes1=int(net_cfg.get("modes1", 12)),
            width=int(net_cfg.get("width", 32)),
            depth=int(net_cfg.get("depth", 4)),
            channels_last_proj=int(net_cfg.get("channels_last_proj", 128)),
            out_channels=int(net_cfg.get("out_channels", 1)),
            padding=int(net_cfg.get("padding", 0)),
        )
    if t == "fno2d":
        from underPINN.nn.operators import FNO2D
        return FNO2D(
            modes1=int(net_cfg.get("modes1", 12)),
            modes2=int(net_cfg.get("modes2", 12)),
            width=int(net_cfg.get("width", 32)),
            depth=int(net_cfg.get("depth", 4)),
            channels_last_proj=int(net_cfg.get("channels_last_proj", 128)),
            out_channels=int(net_cfg.get("out_channels", 1)),
            padding=int(net_cfg.get("padding", 0)),
        )
    if t == "deeponet":
        from underPINN.nn.operators import DeepONet1D
        if "branch_layers" not in net_cfg or "trunk_layers" not in net_cfg:
            raise KeyError("network.type='deeponet' requires 'branch_layers' "
                          "and 'trunk_layers' lists")
        return DeepONet1D(
            branch_layers=list(net_cfg["branch_layers"]),
            trunk_layers=list(net_cfg["trunk_layers"]),
            gated=bool(net_cfg.get("gated", True)),
        )
    if t == "cvit":
        from underPINN.nn.operators import CVit
        return CVit(
            patch_size=tuple(net_cfg.get("patch_size", (1, 16, 16))),
            grid_size=tuple(net_cfg.get("grid_size", (128, 128))),
            latent_dim=int(net_cfg.get("latent_dim", 256)),
            emb_dim=int(net_cfg.get("emb_dim", 256)),
            depth=int(net_cfg.get("depth", 3)),
            num_heads=int(net_cfg.get("num_heads", 8)),
            dec_emb_dim=int(net_cfg.get("dec_emb_dim", 256)),
            dec_num_heads=int(net_cfg.get("dec_num_heads", 8)),
            dec_depth=int(net_cfg.get("dec_depth", 1)),
            num_mlp_layers=int(net_cfg.get("num_mlp_layers", 1)),
            mlp_ratio=int(net_cfg.get("mlp_ratio", 1)),
            out_dim=int(net_cfg.get("out_dim", 1)),
            eps=float(net_cfg.get("eps", 1e5)),
            layer_norm_eps=float(net_cfg.get("layer_norm_eps", 1e-5)),
            embedding_type=str(net_cfg.get("embedding_type", "grid")),
        )
    return MLP(layers=_layers())


def network_config(cfg) -> Dict[str, Any]:
    """Extract the ``network`` config dict from a loaded config object.

    For the temporal-Fourier network the base frequency ω is taken from
    ``network.omega`` if given, else derived as 2π/``physics.T_period``.  The
    returned dict is what should be both fed to :func:`build_model` and saved as
    the run's ``network`` metadata (so prediction rebuilds the same model).
    """
    from underPINN.config.loader import cfg_get

    nc = cfg.network
    out: Dict[str, Any] = {"type": str(cfg_get(nc, "type", default="gated_mlp")).lower()}
    layers = cfg_get(nc, "layers", default=None)      # operators may omit this
    if layers is not None:
        out["layers"] = list(layers)
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
    elif out["type"] == "fno1d":
        out["modes1"]             = int(cfg_get(nc, "modes1", default=12))
        out["width"]              = int(cfg_get(nc, "width", default=32))
        out["depth"]              = int(cfg_get(nc, "depth", default=4))
        out["channels_last_proj"] = int(cfg_get(nc, "channels_last_proj", default=128))
        out["out_channels"]       = int(cfg_get(nc, "out_channels", default=1))
        out["padding"]            = int(cfg_get(nc, "padding", default=0))
    elif out["type"] == "fno2d":
        out["modes1"]             = int(cfg_get(nc, "modes1", default=12))
        out["modes2"]             = int(cfg_get(nc, "modes2", default=12))
        out["width"]              = int(cfg_get(nc, "width", default=32))
        out["depth"]              = int(cfg_get(nc, "depth", default=4))
        out["channels_last_proj"] = int(cfg_get(nc, "channels_last_proj", default=128))
        out["out_channels"]       = int(cfg_get(nc, "out_channels", default=1))
        out["padding"]            = int(cfg_get(nc, "padding", default=0))
    elif out["type"] == "deeponet":
        out["branch_layers"] = list(cfg_get(nc, "branch_layers"))
        out["trunk_layers"]  = list(cfg_get(nc, "trunk_layers"))
        out["gated"]         = bool(cfg_get(nc, "gated", default=True))
    elif out["type"] == "cvit":
        out["patch_size"]     = list(cfg_get(nc, "patch_size", default=[1, 16, 16]))
        out["grid_size"]      = list(cfg_get(nc, "grid_size", default=[128, 128]))
        out["latent_dim"]     = int(cfg_get(nc, "latent_dim", default=256))
        out["emb_dim"]        = int(cfg_get(nc, "emb_dim", default=256))
        out["depth"]          = int(cfg_get(nc, "depth", default=3))
        out["num_heads"]      = int(cfg_get(nc, "num_heads", default=8))
        out["dec_emb_dim"]    = int(cfg_get(nc, "dec_emb_dim", default=256))
        out["dec_num_heads"]  = int(cfg_get(nc, "dec_num_heads", default=8))
        out["dec_depth"]      = int(cfg_get(nc, "dec_depth", default=1))
        out["num_mlp_layers"] = int(cfg_get(nc, "num_mlp_layers", default=1))
        out["mlp_ratio"]      = int(cfg_get(nc, "mlp_ratio", default=1))
        out["out_dim"]        = int(cfg_get(nc, "out_dim", default=1))
        out["eps"]            = float(cfg_get(nc, "eps", default=1e5))
        out["layer_norm_eps"] = float(cfg_get(nc, "layer_norm_eps", default=1e-5))
        out["embedding_type"] = str(cfg_get(nc, "embedding_type", default="grid"))
    return out
