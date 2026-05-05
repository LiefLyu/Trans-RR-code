"""Shared ridge-penalty grids used across all simulation experiments.

Importing this module is the single source of truth. Any caller that needs
a default tau grid should `from transrr_lib._grids import TAU_GRID`.

Grid design (paper revision, May 2026)
--------------------------------------
TAU_GRID and WIDE_TAU_GRID are defined so that ``TAU_GRID`` is a strict
subset of ``WIDE_TAU_GRID`` (same step size 1/2 in log_3, just narrower span).
This makes the §4.5 wide-grid sensitivity check a clean isolation of "do
the additional outer points change anything?".

TAU_GRID (default for all main and sensitivity simulations):
    np.logspace(-2, 2, 9, base=3)
    = 9 points spaced 1/2 in log_3:
      [1/9, 0.193, 1/3, 0.577, 1.0, 1.732, 3.0, 5.196, 9.0]

WIDE_TAU_GRID (used by §4.5 grid-width sensitivity table B4):
    np.logspace(-3, 3, 13, base=3)
    = 13 points spaced 1/2 in log_3:
      adds 2 outer points on each side relative to TAU_GRID, namely
      {1/27, 1/27 * 3^{1/2}} below 1/9 and the symmetric pair above 9,
      capping at 27.

By construction TAU_GRID is a subset of WIDE_TAU_GRID, with 9 shared
points (the entirety of TAU_GRID).

Choice rationale (summary; full diagnostic in commit messages):

(a) ``diagnose_tau_grid.py`` (boundary-hit rates of CV-selected tau,
    K = 200 reps over 3 cases x 3 discrepancy levels x 4 ridge-penalty
    roles): aggregate median 1.7, IQR contained in [0.7, 5.2]. Boundary
    hits at the lower end of WIDE_TAU_GRID below 1% in all 9 cells.
    Boundary hits at the upper end of WIDE_TAU_GRID below 5% in 8 of 9
    cells, with the cauchy/h=1/trans_source cell hitting 27 in 43.5%
    of replications.

(b) ``diagnose_oracle_vs_cv.py`` (CV-selected tau vs oracle tau that
    minimises ||what(tau) - w_true||^2 on the FULL data, Cauchy case
    only, K = 200): the oracle never selects tau >= 27 in any cell.
    Median Trans-RR regret err(tau_CV) / err(tau_oracle) below 2.2% in
    every (h, method) cell. So the upper boundary hit is CV finite-sample
    noise on a flat plateau of the error curve.

REALDATA_TAU_GRID rationale
---------------------------
The real-data NIR predictors are not on a unit-variance scale, so the
simulation TAU_GRID does not transfer directly. We retain the existing
realdata grid ``np.logspace(-4, 1, 11)`` (base 10) calibrated against
the realdata pipeline runs.
"""
import numpy as np

#: Default tau grid for §4.3 / §4.4 / §4.5 simulation experiments.
#: 9 points, geometrically spaced from 1/9 to 9, step 1/2 in log_3.
TAU_GRID = np.logspace(-2, 2, 9, base=3)

#: Wider tau grid used by the §4.5 grid-width sensitivity check (Tab B4).
#: 13 points, geometrically spaced from 1/27 to 27, step 1/2 in log_3.
#: Strict superset of TAU_GRID (same step in log_3, extends 2 points
#: lower and 2 points higher).
WIDE_TAU_GRID = np.logspace(-3, 3, 13, base=3)

#: Tau grid for §5 real-data experiments (different scale, not unified
#: with the simulation grid).
REALDATA_TAU_GRID = np.logspace(-4, 1, 11)
