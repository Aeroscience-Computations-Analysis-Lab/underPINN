# Model Checkpointing & Inference

Every runner writes two files to the output directory after training completes:

```text
outputs/burgers/
  params.msgpack       ← exact Flax/msgpack serialization of all weights
  params_meta.json     ← {"problem": "burgers", "network": {"type": "mlp", "layers": [...]}, ...}
  predictions.npz      ← collocation-point predictions
  config.yaml          ← resolved training config (reproducibility)
  loss_hist.npy
  loss.png
```

```{note}
This is distinct from the `<out_dir>/restart/` snapshot described in {doc}`restart`,
which tracks in-progress training state (optimizer moments, epoch counter) for
fault-tolerant resumption.
```

## Save during training — `ModelCheckpoint` callback

```python
from underPINN.callbacks.checkpoint import ModelCheckpoint

ModelCheckpoint(
    out_dir="outputs/burgers/",
    monitor="loss",
    mode="min",
    save_best_only=True,
    metadata={
        "problem": "burgers",
        "network": {"type": "mlp", "layers": [2, 64, 64, 64, 1]},
    },
)
```

## Reload and predict on new inputs

```python
from underPINN.utils.checkpoint import ModelPredictor
import jax.numpy as jnp

# Option A — auto-build model from saved metadata (zero boilerplate)
predictor = ModelPredictor.from_meta("outputs/burgers/")

# Option B — provide the model explicitly
from underPINN.nn.mlp import MLP
predictor = ModelPredictor.from_checkpoint(
    MLP(layers=[2, 64, 64, 64, 1]),
    "outputs/burgers/",
)

# Run inference
x_new = jnp.linspace(-1.0, 1.0, 500)
t_new = jnp.full(500, 0.8)
u = predictor.predict(jnp.stack([x_new, t_new], axis=1))
```

```{tip}
`ModelPredictor.from_meta` reads `params_meta.json` and rebuilds the **exact**
architecture used at training time — you never need to re-specify layer widths or
network type by hand.
```

## Lower-level API

```python
from underPINN.utils.checkpoint import save_checkpoint, load_checkpoint

# Save any param pytree
save_checkpoint(params, "my_dir/", metadata={"problem": "wave", "network": {"layers": [...]}})

# Load (model used as template for structure)
params = load_checkpoint(model, "my_dir/")
```

## Post-processing utilities

```bash
# Steady pipe & AAA (Newtonian or Carreau) — axial-plane u contour + streamlines,
# pressure contour & line plots, wall shear stress, and an NPZ of the solution
python examples/predict_steady.py outputs/pipe_flow
python examples/predict_steady.py outputs/AAA_rheology

# Pulsatile pipe (time-marching) — point queries, snapshot/spacetime plots, GIF
python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer --t 2.7 --plot
python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer --spacetime --animate
```
