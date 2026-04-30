"""
Adaptive aggregation of Trans-RR and Single-RR via CV-selected weight theta.

Defines beta_ada = theta * beta_trans + (1 - theta) * beta_single,
with theta chosen on a grid by minimizing the held-out validation loss
of the convex combination on a K-fold partition that is INDEPENDENT
of the partition used to tune the base estimators' ridge penalties
(see Algorithm 2 in the paper).

When theta = 1 -> beta_trans (the original Trans-RR).
When theta = 0 -> beta_single.
The grid includes 0 and 1, so the procedure is at least as good as
choosing the better of the two base estimators in the asymptotic
sense.
"""

import numpy as np


def aggregate_by_cv(
    fold_indices,
    fold_betas_trans,
    fold_betas_single,
    X, Y,
    beta_trans_full,
    beta_single_full,
    theta_grid=None,
    criterion='mae',
):
    """
    Pick the convex-combination weight theta by K-fold CV on the partition
    described by fold_indices.

    The fold-level base estimators fold_betas_trans[k] and fold_betas_single[k]
    must be computed on the SAME training portion (i.e. fold_indices[k]'s
    train_idx); otherwise the validation residual on the corresponding val_idx
    is not well-defined for the convex combination.

    fold_indices may be drawn independently of any partition used to tune the
    base estimators' ridge penalties; the caller is responsible for refitting
    the fold-level base estimators on the partition described here.

    Parameters
    ----------
    fold_indices : list of (train_idx, val_idx) pairs
        The fold partition. Same partition used to construct both
        fold_betas_trans and fold_betas_single below.
    fold_betas_trans : list of np.ndarray, length n_folds
        Trans-RR fitted on each fold's train_idx at the tuned tau_opt_trans.
    fold_betas_single : list of np.ndarray, length n_folds
        Single-RR fitted on each fold's train_idx at the tuned tau_opt_single.
    X : (n, p) np.ndarray
        Full target training design matrix.
    Y : (n,) np.ndarray
        Full target training response.
    beta_trans_full : (p,) np.ndarray
        Final Trans-RR estimator on the full target sample at tau_opt_trans.
    beta_single_full : (p,) np.ndarray
        Final Single-RR estimator on the full target sample at tau_opt_single.
    theta_grid : 1D array-like or None
        Candidate theta values in [0, 1]. None -> np.linspace(0, 1, 11).
    criterion : {'mae', 'mse'}
        Validation criterion. Should match the one used to tune the base
        estimators (default 'mae').

    Returns
    -------
    dict with keys:
        - 'beta_ada'    : (p,) final aggregated estimator at theta_star
        - 'theta_star'  : selected weight in [0, 1]
        - 'cv_curve'    : (len(theta_grid),) CV losses, one per grid point
        - 'theta_grid'  : the grid used
        - 'cv_loss_min' : minimum CV loss (for diagnostics or auto-selection
                          against another method)
    """
    if criterion not in ('mae', 'mse'):
        raise ValueError(f"criterion must be 'mae' or 'mse', got {criterion!r}")
    if len(fold_betas_trans) != len(fold_betas_single):
        raise ValueError("fold_betas_trans and fold_betas_single must have the same length.")
    if len(fold_indices) != len(fold_betas_trans):
        raise ValueError("fold_indices and fold_betas_trans must have the same length.")

    if theta_grid is None:
        theta_grid = np.linspace(0.0, 1.0, 11)
    theta_grid = np.asarray(theta_grid, dtype=float)

    n_thetas = len(theta_grid)
    cv_curve = np.empty(n_thetas)

    for j, theta in enumerate(theta_grid):
        residuals_concat = []
        for (_, val_idx), beta_t, beta_s in zip(fold_indices, fold_betas_trans, fold_betas_single):
            beta_mix = theta * beta_t + (1.0 - theta) * beta_s
            r = Y[val_idx] - X[val_idx] @ beta_mix
            if criterion == 'mae':
                residuals_concat.append(np.abs(r))
            else:
                residuals_concat.append(r ** 2)
        cv_curve[j] = np.mean(np.concatenate(residuals_concat))

    j_star = int(np.argmin(cv_curve))
    theta_star = float(theta_grid[j_star])
    beta_ada = theta_star * beta_trans_full + (1.0 - theta_star) * beta_single_full

    return {
        'beta_ada': beta_ada,
        'theta_star': theta_star,
        'cv_curve': cv_curve,
        'theta_grid': theta_grid,
        'cv_loss_min': float(cv_curve[j_star]),
    }


def combine(beta_trans, beta_single, theta):
    """Pure utility: convex combination at a fixed theta."""
    return theta * beta_trans + (1.0 - theta) * beta_single
