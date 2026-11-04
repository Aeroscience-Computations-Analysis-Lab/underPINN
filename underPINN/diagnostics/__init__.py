"""Method-agnostic training and network diagnostics for PINNs.

Attach to any solver by building a ``Problem`` (loss terms on a fixed batch, plus optionally the
residual, a loss-as-sum-of-squares vector, the network output and a hidden-activation
function), then:

    mon = TrainingMonitor(problem, params0, every=200)      # one call per training step
    res = run_checkpoint(problem, params, params0=params0)  # Hessian, NTK, slices, representation, refit

See core.py for the Problem contract and each module for its definitions, which follow the
companion notes: "Representation and the Geometry of the PINN Loss" (pinn_math_notes),
"Representation, Operator and the PINN Loss Landscape" (poisson notes) and "Post-processing a
Two-Layer PINN" (twolayer notes).
"""
from .core import Problem, default_group, flax_hidden_fn, param_groups
from .curvature import (effective_dimension, full_hessian, full_hessian_spectrum, gauss_newton_fidelity,
                        jacobi_scaled_spectrum, kappa_top, lanczos, layer_sharpness, sharpness,
                        spectrum_slq, top_eigenvalues)
from .design import offgrid_residual_ratio, output_layer_refit, refit_pruning
from .kernel import ntk, ntk_drift, residual_jacobian, spectral_rank
from .landscape import count_extrema, filter_normalised_direction, slice_1d, slice_2d
from .modal import (laplacian_mode_weights, modal_error, modal_kernel, sine_coefficients,
                    sine_projection)
from .monitor import TrainingMonitor, best_so_far, iterations_to
from .report import Config, run_checkpoint
from .representation import (ablation_importance, erank, feature_gram, hidden_stats, linear_cka,
                             linear_probe, saturation, singular_values)
from .spectral import amplitude_spectrum, energy_fraction_below, grid_step

__all__ = [
    # core
    "Problem", "default_group", "flax_hidden_fn", "param_groups",
    # curvature
    "effective_dimension", "full_hessian", "full_hessian_spectrum", "gauss_newton_fidelity",
    "jacobi_scaled_spectrum", "kappa_top", "lanczos", "layer_sharpness", "sharpness", "spectrum_slq",
    "top_eigenvalues",
    # design
    "offgrid_residual_ratio", "output_layer_refit", "refit_pruning",
    # kernel
    "ntk", "ntk_drift", "residual_jacobian", "spectral_rank",
    # landscape
    "count_extrema", "filter_normalised_direction", "slice_1d", "slice_2d",
    # modal
    "laplacian_mode_weights", "modal_error", "modal_kernel", "sine_coefficients", "sine_projection",
    # monitor
    "TrainingMonitor", "best_so_far", "iterations_to",
    # report
    "Config", "run_checkpoint",
    # representation
    "ablation_importance", "erank", "feature_gram", "hidden_stats", "linear_cka", "linear_probe",
    "saturation", "singular_values",
    # spectral
    "amplitude_spectrum", "energy_fraction_below", "grid_step",
]
