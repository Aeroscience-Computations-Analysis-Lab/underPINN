import jax
import jax.numpy as jnp
from flax import linen as nn
from .subdomain import SubdomainNetwork
from .attention import SimpleGate


def window_1d(x, xmin, xmax, smin, smax):
    return jnp.clip(jax.nn.sigmoid((x-xmin)/smin),1e-10,1.0) * \
           jnp.clip(jax.nn.sigmoid((xmax-x)/smax),1e-10,1.0)

def window_nd(x, xmin, xmax, smin, smax):
    w = 1.0
    for d in range(x.shape[1]):
        w *= window_1d(x[:, d], xmin[d], xmax[d], smin[d], smax[d])
    return w[:, None]

class FBPINN(nn.Module):
    layers: list
    shifts: jnp.ndarray
    xs_min: jnp.ndarray
    xs_max: jnp.ndarray
    smins: jnp.ndarray
    smaxs: jnp.ndarray
    attention_cls: callable = SimpleGate
    # Optional output transform applied inside every subdomain network.
    # Pass a callable  f(x) -> x  to enforce positivity, scaling, etc.
    # Example:  out_transform=k_eps_positivity  for the k-ε turbulence model.
    out_transform: callable = None

    def setup(self):
        self.subnets = [
            SubdomainNetwork(self.layers,
                             attention_cls=self.attention_cls,
                             out_transform=self.out_transform)
            for _ in range(self.shifts.shape[0])
        ]

    def __call__(self, x):
        out = 0.0
        for i, net in enumerate(self.subnets):
            win = window_nd(x, self.xs_min[i], self.xs_max[i], self.smins[i], self.smaxs[i])
            out += net(x - self.shifts[i]) * win
        return out


# ---------------------------------------------------------------------------
# Stacked FBPINN -- the same model, laid out for model (subdomain) parallelism
# ---------------------------------------------------------------------------
#
# FBPINN keeps every subdomain network as a separate module (subnets_0, ...)
# evaluated in a Python loop. That layout cannot be split across devices: JAX
# shards *arrays*, not lists of modules. StackedFBPINN holds the identical
# networks with their weights stacked along a leading "subdomain" axis and
# evaluates them with one vmap, so that axis can be sharded across devices --
# each device then evaluates only its own subdomains, and XLA sums the
# contributions (see underPINN.training.model_parallel).
#
# It is the same function: outputs, derivatives and gradients match FBPINN to
# float32 rounding (the sum over subdomains is merely reordered), and init
# reproduces FBPINN's random initialisation bit-for-bit, so switching layouts
# does not change a training run.

def _pad_rows(a, n_pad: int):
    """Append *n_pad* copies of row 0 (dummy subdomains; masked out)."""
    if n_pad == 0:
        return a
    return jnp.concatenate([a, jnp.repeat(a[:1], n_pad, axis=0)], axis=0)


def stack_fbpinn_params(params, n_pad: int = 0) -> dict:
    """Convert FBPINN params (``subnets_0 ... subnets_{S-1}``) to the stacked
    layout used by :class:`StackedFBPINN`, appending *n_pad* masked dummy
    subdomains (copies of subdomain 0)."""
    inner = params["params"]
    keys = sorted((k for k in inner if k.startswith("subnets_")),
                  key=lambda k: int(k.rsplit("_", 1)[1]))
    if not keys or len(keys) != len(inner):
        raise ValueError(
            "expected FBPINN params of the form {'params': {'subnets_0': ..., "
            f"'subnets_1': ...}}; got top-level keys {sorted(inner)}")
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a),
                                     *[inner[k] for k in keys])
    stacked = jax.tree_util.tree_map(lambda a: _pad_rows(a, n_pad), stacked)
    return {"params": {"subnets": stacked}}


def unstack_fbpinn_params(params, n_pad: int = 0) -> dict:
    """Inverse of :func:`stack_fbpinn_params`: back to the FBPINN layout,
    dropping the last *n_pad* (dummy) subdomains. Use it to save a checkpoint
    from a model-parallel run in the ordinary FBPINN format."""
    sub = params["params"]["subnets"]
    total = jax.tree_util.tree_leaves(sub)[0].shape[0]
    return {"params": {f"subnets_{i}": jax.tree_util.tree_map(lambda a: a[i], sub)
                       for i in range(total - n_pad)}}


class StackedFBPINN(nn.Module):
    """:class:`FBPINN` with subdomain weights stacked for model parallelism.

    Same fields as :class:`FBPINN`, plus ``n_pad``: the number of trailing
    *dummy* subdomains appended so the subdomain count divides the number of
    devices. Dummy subdomains are multiplied by an exact 0 mask, so they add
    nothing to the output and receive exactly zero gradient (an optimizer such
    as Adam therefore never moves them).

    Build one from an existing model with :meth:`from_fbpinn` rather than
    directly.
    """
    layers: list
    shifts: jnp.ndarray
    xs_min: jnp.ndarray
    xs_max: jnp.ndarray
    smins: jnp.ndarray
    smaxs: jnp.ndarray
    attention_cls: callable = SimpleGate
    out_transform: callable = None
    n_pad: int = 0

    def setup(self):
        Stacked = nn.vmap(SubdomainNetwork, variable_axes={"params": 0},
                          split_rngs={"params": True}, in_axes=0, out_axes=0)
        self.subnets = Stacked(self.layers, attention_cls=self.attention_cls,
                               out_transform=self.out_transform)

    def __call__(self, x):
        n_total = self.shifts.shape[0]
        outs = self.subnets(x[None, :, :] - self.shifts[:, None, :])  # (S, N, O)
        wins = jax.vmap(window_nd, in_axes=(None, 0, 0, 0, 0))(
            x, self.xs_min, self.xs_max, self.smins, self.smaxs)       # (S, N, 1)
        contrib = outs * wins
        if self.n_pad:
            mask = (jnp.arange(n_total) < n_total - self.n_pad).astype(contrib.dtype)
            contrib = contrib * mask[:, None, None]
        return jnp.sum(contrib, axis=0)

    @property
    def n_subdomains(self) -> int:
        """Real (non-dummy) subdomain count."""
        return self.shifts.shape[0] - self.n_pad

    @nn.nowrap
    def to_fbpinn(self) -> "FBPINN":
        """The equivalent :class:`FBPINN` (dummy subdomains dropped)."""
        n = self.n_subdomains
        return FBPINN(layers=self.layers, shifts=self.shifts[:n],
                      xs_min=self.xs_min[:n], xs_max=self.xs_max[:n],
                      smins=self.smins[:n], smaxs=self.smaxs[:n],
                      attention_cls=self.attention_cls,
                      out_transform=self.out_transform)

    @nn.nowrap
    def init(self, rngs, *args, **kwargs):
        """Initialise exactly as the equivalent :class:`FBPINN` would.

        ``nn.vmap`` would otherwise draw different random weights, so a run
        would silently change just by switching layouts. (``nn.nowrap`` keeps
        Flax from treating this as a module method, which would register the
        reference FBPINN as a child module.)
        """
        ref = self.to_fbpinn().init(rngs, *args, **kwargs)
        return stack_fbpinn_params(ref, n_pad=self.n_pad)

    @nn.nowrap
    def export_params(self, params) -> dict:
        """*params* in the ordinary FBPINN layout, ready to save or to load
        into :meth:`to_fbpinn`'s model."""
        return unstack_fbpinn_params(params, n_pad=self.n_pad)

    @classmethod
    def from_fbpinn(cls, model: "FBPINN", multiple: int = 1) -> "StackedFBPINN":
        """Stacked equivalent of *model*, padded so its subdomain count is a
        multiple of *multiple* (the number of devices to split across)."""
        if isinstance(model, StackedFBPINN):
            model = model.to_fbpinn()
        if not isinstance(model, FBPINN):
            raise TypeError(f"expected an FBPINN, got {type(model).__name__}")
        if multiple < 1:
            raise ValueError(f"multiple must be >= 1, got {multiple}")
        n_pad = (-model.shifts.shape[0]) % multiple
        return cls(layers=model.layers,
                   shifts=_pad_rows(model.shifts, n_pad),
                   xs_min=_pad_rows(model.xs_min, n_pad),
                   xs_max=_pad_rows(model.xs_max, n_pad),
                   smins=_pad_rows(model.smins, n_pad),
                   smaxs=_pad_rows(model.smaxs, n_pad),
                   attention_cls=model.attention_cls,
                   out_transform=model.out_transform,
                   n_pad=n_pad)
