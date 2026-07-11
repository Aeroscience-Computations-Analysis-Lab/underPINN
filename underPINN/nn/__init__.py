"""underPINN.nn — neural-network architectures and the model factory.

``build_model`` / ``network_config`` are the single model-building path used by
the examples: new architectures (including neural operators) register once in
:mod:`underPINN.nn.factory` and become selectable from any case's YAML via
``network.type``.
"""
from underPINN.nn.factory import build_model, network_config
from underPINN.nn.mlp import (
    MLP,
    GatedMLP,
    FourierMLP,
    TemporalFourierMLP,
    SIREN,
)
from underPINN.nn.fbpinn import FBPINN

__all__ = [
    "build_model",
    "network_config",
    "MLP",
    "GatedMLP",
    "FourierMLP",
    "TemporalFourierMLP",
    "SIREN",
    "FBPINN",
]
