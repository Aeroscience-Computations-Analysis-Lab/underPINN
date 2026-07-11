"""underPINN.pde — physics residual operators (PDEs and ODEs).

Each module defines one or more :class:`underPINN.core.base.BasePDE`
subclasses.  Import the specific module you need, e.g.::

    from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE

(No eager re-exports here: PDE modules are imported lazily so that importing
one physics family does not pull in every other one.)
"""
