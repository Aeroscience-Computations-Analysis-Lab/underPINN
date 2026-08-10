"""underPINN.solver — training-loop orchestrators.

Import the specific solver you need, e.g.::

    from underPINN.solver.fbpinn import FBPINNSolver
    from underPINN.solver.ode_solver import ODESolver

(No eager re-exports: solvers are imported lazily to keep package import
light.)
"""
