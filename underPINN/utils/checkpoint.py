"""Model checkpoint utilities — save, load, and run inference.

Quick start
-----------
**Save** (inside a runner after training)::

    from underPINN.utils.checkpoint import save_checkpoint
    save_checkpoint(solver.params, out_dir,
                    metadata={"problem": "burgers",
                              "network": {"type": "mlp", "layers": [2,64,64,64,1]}})
    # writes:  out_dir/params.msgpack
    #          out_dir/params_meta.json

**Load & predict** (in a separate inference script)::

    from underPINN.nn.mlp import MLP
    from underPINN.utils.checkpoint import ModelPredictor
    import jax.numpy as jnp

    predictor = ModelPredictor.from_checkpoint(
        MLP(layers=[2, 64, 64, 64, 1]),
        "outputs/burgers/",
    )
    u = predictor.predict(jnp.stack([x, t], axis=1))

Or at a lower level::

    from underPINN.utils.checkpoint import load_checkpoint
    params = load_checkpoint(model, "outputs/burgers/")
    u = model.apply(params, inputs)

Checkpoint format
-----------------
``params.msgpack``   — Flax-serialized parameter pytree (binary, exact round-trip)
``params_meta.json`` — JSON sidecar with network architecture and any extra metadata
                       (human-readable; optional but strongly recommended)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _msgpack_path(base: str | Path, stem: str = "params") -> Path:
    base = Path(base)
    if base.suffix == ".msgpack":
        return base
    return base / f"{stem}.msgpack"


def _meta_path(msgpack_path: Path) -> Path:
    return msgpack_path.with_suffix("").with_suffix(".json").with_name(
        msgpack_path.stem + "_meta.json"
    )


def _flax_to_bytes(params) -> bytes:
    try:
        from flax import serialization
        return serialization.to_bytes(params)
    except ImportError as e:
        raise ImportError(
            "Flax is required for checkpoint serialization. "
            "Install it with:  pip install flax"
        ) from e


def _flax_from_bytes(template, data: bytes):
    from flax import serialization
    return serialization.from_bytes(template, data)


# ---------------------------------------------------------------------------
# Public save / load API
# ---------------------------------------------------------------------------

def save_checkpoint(
    params,
    out_dir,
    stem: str = "params",
    metadata: Optional[dict] = None,
):
    """Serialize *params* to disk as a Flax msgpack checkpoint.

    Parameters
    ----------
    params   : Flax parameter pytree (e.g. ``solver.params``).
    out_dir  : Directory to write into (created if absent).
    stem     : Base filename without extension (default ``"params"``).
               The files written are ``<stem>.msgpack`` and
               ``<stem>_meta.json``.
    metadata : Optional dict stored alongside the checkpoint as JSON.
               Recommended keys: ``"problem"``, ``"network"``
               (with ``"type"`` and ``"layers"``).  Any JSON-serialisable
               values are accepted.

    Returns
    -------
    (params_path, meta_path)
        Absolute paths of the two files created.
        *meta_path* is ``None`` when *metadata* is not provided.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp = out_dir / f"{stem}.msgpack"
    with open(mp, "wb") as f:
        f.write(_flax_to_bytes(params))
    print(f"Checkpoint saved → {mp}")

    meta_p = None
    if metadata is not None:
        meta_p = out_dir / f"{stem}_meta.json"
        with open(meta_p, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata   saved → {meta_p}")

    return str(mp), str(meta_p) if meta_p else None


def load_checkpoint(model_or_template, path):
    """Deserialize a Flax checkpoint from *path*.

    Parameters
    ----------
    model_or_template : Either

      * a Flax module (the function calls ``model.init`` with a dummy
        input to obtain the parameter template), **or**
      * an already-initialised parameter pytree used directly as the
        template for ``flax.serialization.from_bytes``.

    path : Either

      * a directory — the function looks for ``params.msgpack`` inside, **or**
      * the direct path to a ``.msgpack`` file.

    Returns
    -------
    params
        The loaded Flax parameter pytree.
    """
    mp = _msgpack_path(path)
    if not mp.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {mp}.\n"
            f"Make sure you saved the model with save_checkpoint() first."
        )

    with open(mp, "rb") as f:
        data = f.read()

    # Obtain a template pytree
    import jax.numpy as jnp
    template = model_or_template
    if hasattr(model_or_template, "init"):
        # It's a Flax module — initialise it to get the structure
        import jax
        # Infer input dimension from metadata if possible
        meta = read_metadata(path)
        in_features = (meta.get("network", {}).get("layers", [2])[0]
                       if meta else 2)
        key    = jax.random.PRNGKey(0)
        template = model_or_template.init(key, jnp.ones((1, in_features)))

    return _flax_from_bytes(template, data)


def read_metadata(path) -> Optional[dict]:
    """Read the JSON sidecar saved alongside a checkpoint.

    Returns ``None`` if no sidecar exists.
    """
    mp   = _msgpack_path(path)
    meta = _meta_path(mp)
    if not meta.exists():
        return None
    with open(meta) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ModelPredictor — one-stop inference object
# ---------------------------------------------------------------------------

class ModelPredictor:
    """Wraps a model + loaded parameters for easy inference.

    Usage::

        predictor = ModelPredictor.from_checkpoint(
            MLP(layers=[2, 64, 64, 64, 1]),
            "outputs/burgers/",
        )
        u = predictor.predict(jnp.stack([x, t], axis=1))

    The predictor is also callable::

        u = predictor(jnp.stack([x, t], axis=1))
    """

    def __init__(self, model, params):
        self.model  = model
        self.params = params

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, model, path: str | Path) -> "ModelPredictor":
        """Load params from *path* using *model* as the template.

        Parameters
        ----------
        model : Flax module (already instantiated, e.g. ``MLP(layers=…)``).
        path  : Directory or ``.msgpack`` file path.
        """
        params = load_checkpoint(model, path)
        return cls(model, params)

    @classmethod
    def from_meta(cls, path: str | Path) -> "ModelPredictor":
        """Auto-build the model from saved metadata, then load params.

        Requires that ``params_meta.json`` was written alongside the
        checkpoint (i.e. ``metadata`` was passed to :func:`save_checkpoint`).
        The JSON must contain a ``"network"`` key with at least ``"layers"``.
        """
        meta = read_metadata(path)
        if meta is None:
            raise FileNotFoundError(
                f"No metadata sidecar found at {_meta_path(_msgpack_path(path))}.\n"
                f"Pass metadata when saving or use from_checkpoint() with an "
                f"explicit model."
            )

        net = meta.get("network", {})
        layers = net.get("layers")
        if layers is None:
            raise ValueError("Metadata 'network.layers' is required for auto-rebuild.")

        net_type = net.get("type", "mlp").lower()
        from underPINN.nn.mlp import MLP, FourierMLP
        if net_type == "fourier_mlp":
            model = FourierMLP(
                layers=layers,
                n_fourier=net.get("n_fourier", 16),
                sigma=net.get("sigma", 2.0),
            )
        else:
            model = MLP(layers=layers)

        return cls.from_checkpoint(model, path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, inputs):
        """Run model forward pass.  Returns raw JAX array (shape depends on model)."""
        return self.model.apply(self.params, inputs)

    def __call__(self, inputs):
        return self.predict(inputs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_numpy(self, inputs) -> np.ndarray:
        """Like :meth:`predict` but returns a NumPy array."""
        return np.array(self.predict(inputs))

    def __repr__(self) -> str:
        return (f"ModelPredictor(model={self.model.__class__.__name__}, "
                f"params_leaves={len(list(self.params))!r})")
