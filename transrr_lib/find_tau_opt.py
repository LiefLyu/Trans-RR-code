import numpy as np
from sklearn.model_selection import KFold

from ._grids import TAU_GRID
from .robust_ridge_optimizer import solve_robust_ridge


def find_optimal_tau_robust_ridge(
    X, Y,
    n_folds=5,
    tau_range=None,
    psi_delta=1.35,
    psi_eta=0.1,
    criterion='mae',
    loss='smoothed_huber',
    return_folds=False,
):
    """
    5-fold CV over a tau grid for the robust-ridge regression.

    By default returns the scalar tau_opt for backward compatibility.
    With return_folds=True, returns a dict that also exposes per-fold
    estimates and CV diagnostics, used by the Adaptive aggregation layer.

    Parameters
    ----------
    X : (n, p) np.ndarray
    Y : (n,) np.ndarray
    n_folds : int, default 5
    tau_range : 1D array-like or None
        Candidate tau values. None -> default ``transrr_lib._grids.TAU_GRID``.
    psi_delta : float
        delta in the robust loss (paper default 1.35).
    psi_eta : float
        eta in the smoothed Huber loss (paper default 0.1). Ignored when loss='pseudo_huber'.
    criterion : {'mae', 'mse'}
        Validation criterion for the inner CV. Default 'mae' matches the paper.
    loss : {'smoothed_huber', 'pseudo_huber'}
        Robust loss family. Passed through to solve_robust_ridge.
    return_folds : bool
        If False (default), returns the scalar tau_opt.
        If True, returns a dict with keys:
            - 'tau_opt'       : selected tau (scalar)
            - 'cv_loss_min'   : min cross-validation loss over the grid
            - 'cv_curve'      : (len(tau_range),) array of CV losses per tau
            - 'tau_range'     : the grid actually used (1D ndarray)
            - 'fold_indices'  : list of (train_idx, val_idx) pairs (length n_folds)
            - 'fold_betas'    : list of length n_folds; each entry is the fitted
                                beta at tau_opt on that fold's training portion.
                                Shape per entry: (p,) ndarray.

    Returns
    -------
    tau_opt or dict (see return_folds)
    """
    if criterion not in ('mae', 'mse'):
        raise ValueError(f"criterion must be 'mae' or 'mse', got {criterion!r}")
    if loss not in ('smoothed_huber', 'pseudo_huber'):
        raise ValueError(f"loss must be 'smoothed_huber' or 'pseudo_huber', got {loss!r}")

    if tau_range is None:
        tau_range = TAU_GRID
    tau_range = np.asarray(tau_range, dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    fold_indices = list(kf.split(X))

    n_taus = len(tau_range)
    cv_errors = np.zeros(n_taus)

    # Store per-fold betas for every tau if return_folds; else None
    fold_betas_grid = (
        [[None] * n_folds for _ in range(n_taus)] if return_folds else None
    )

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        for i, tau in enumerate(tau_range):
            try:
                initial_guess = np.linalg.solve(
                    X_train.T @ X_train / len(Y_train) + tau * np.eye(X_train.shape[1]),
                    X_train.T @ Y_train / len(Y_train),
                )
                delta_hat = solve_robust_ridge(
                    X_train, Y_train, tau, psi_delta, psi_eta,
                    initial_beta=initial_guess, loss=loss,
                )
                if return_folds:
                    fold_betas_grid[i][fold_idx] = delta_hat

                residuals = Y_val - X_val @ delta_hat
                if criterion == 'mae':
                    val_error = np.mean(np.abs(residuals))
                else:
                    val_error = np.mean(residuals ** 2)
                cv_errors[i] += val_error / n_folds
            except Exception as e:
                cv_errors[i] += np.inf
                print(f"Warning: Optimization failed for tau={tau} in fold {fold_idx}. Error: {e}")

    best_tau_idx = int(np.argmin(cv_errors))
    tau_opt = float(tau_range[best_tau_idx])

    if not return_folds:
        return tau_opt

    return {
        'tau_opt': tau_opt,
        'cv_loss_min': float(cv_errors[best_tau_idx]),
        'cv_curve': cv_errors,
        'tau_range': tau_range,
        'fold_indices': fold_indices,
        'fold_betas': fold_betas_grid[best_tau_idx],  # per-fold beta at tau_opt
    }
