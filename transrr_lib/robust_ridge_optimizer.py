# robust_ridge_optimizer.py

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Smoothed Huber loss (default; matches paper's eq:rho_eta)
# ============================================================
def _rho_eta_scalar(x, delta, eta):
    """Scalar version of the smoothed Huber loss."""
    c_rho = -eta ** 2 / 6 + eta * delta / 2 - delta ** 2 / 2
    abs_x = np.abs(x)
    if abs_x <= delta - eta:
        val = x ** 2 / 2
    elif (delta - eta) < abs_x < delta:
        val = (delta - eta / 2) * abs_x + (delta - abs_x) ** 3 / (6 * eta) + c_rho
    else:
        val = (delta - eta / 2) * abs_x + c_rho
    return val


def _psi_eta_scalar(x, delta, eta):
    """Scalar version of psi_eta = derivative of smoothed Huber."""
    abs_x = np.abs(x)
    if abs_x <= delta - eta:
        val = x
    elif (delta - eta) < abs_x < delta:
        val = np.sign(x) * (delta - eta / 2 - (delta - abs_x) ** 2 / (2 * eta))
    else:
        val = np.sign(x) * (delta - eta / 2)
    return val


_rho_eta_vec = np.vectorize(_rho_eta_scalar, excluded=['delta', 'eta'])
_psi_eta_vec = np.vectorize(_psi_eta_scalar, excluded=['delta', 'eta'])


# ============================================================
# Pseudo-Huber loss (alternative; analytic, no eta needed)
#   rho_PH(t; delta) = delta^2 * (sqrt(1 + (t/delta)^2) - 1)
#   psi_PH(t; delta) = t / sqrt(1 + (t/delta)^2)
# ============================================================
def _rho_pseudo_huber(x, delta):
    return delta ** 2 * (np.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def _psi_pseudo_huber(x, delta):
    return x / np.sqrt(1.0 + (x / delta) ** 2)


# ============================================================
# Objective and gradient (loss-agnostic dispatcher)
# ============================================================
def _eval_rho_psi(residuals, loss, delta, eta):
    """Return (rho_values, psi_values) for the chosen loss."""
    if loss == 'smoothed_huber':
        return (
            _rho_eta_vec(residuals, delta=delta, eta=eta),
            _psi_eta_vec(residuals, delta=delta, eta=eta),
        )
    elif loss == 'pseudo_huber':
        return (
            _rho_pseudo_huber(residuals, delta),
            _psi_pseudo_huber(residuals, delta),
        )
    else:
        raise ValueError(f"loss must be 'smoothed_huber' or 'pseudo_huber', got {loss!r}")


def _objective_function(beta, x, y, tau, delta, eta, loss):
    n = len(y)
    if n == 0:
        return (tau / 2) * np.sum(beta ** 2)
    residuals = y - x @ beta
    rho_vals, _ = _eval_rho_psi(residuals, loss, delta, eta)
    return (1 / n) * np.sum(rho_vals) + (tau / 2) * np.sum(beta ** 2)


def _gradient_function(beta, x, y, tau, delta, eta, loss):
    n = len(y)
    if n == 0:
        return tau * beta
    residuals = y - x @ beta
    _, psi_vals = _eval_rho_psi(residuals, loss, delta, eta)
    grad_loss = -(1 / n) * x.T @ psi_vals
    return grad_loss + tau * beta


# ============================================================
# Public solver
# ============================================================
def solve_robust_ridge(x, y, tau, delta, eta, initial_beta=None, tol=1e-6, max_iter=10000, loss='smoothed_huber'):
    """
    Solve  argmin_{beta} [ (1/n) sum_i rho(y_i - x_i^T beta) + (tau/2) ||beta||^2 ]
    via L-BFGS-B with analytic gradient.

    Parameters
    ----------
    x : (n, p) numpy.ndarray
    y : (n,) numpy.ndarray
    tau : float >= 0
        L2 regularization strength.
    delta : float > 0
        Loss-shape parameter (paper default 1.35).
    eta : float > 0, eta <= delta
        Smoothing parameter for the smoothed Huber loss. Ignored when
        loss='pseudo_huber'.
    initial_beta : (p,) numpy.ndarray or None
        Initial guess. Pass the closed-form ridge solution for reliable convergence.
    tol : float
        Gradient-norm tolerance for L-BFGS-B.
    max_iter : int
        Max iterations.
    loss : {'smoothed_huber', 'pseudo_huber'}
        Robust loss family. Default 'smoothed_huber' (paper's eq:rho_eta).
        Pseudo-Huber is a C^infty alternative used for sensitivity analysis.

    Returns
    -------
    beta_hat : (p,) numpy.ndarray
    """
    if not (isinstance(x, np.ndarray) and x.ndim == 2):
        raise ValueError("X must be a 2D numpy array.")
    if not (isinstance(y, np.ndarray) and y.ndim == 1):
        raise ValueError("y must be a 1D numpy array.")
    if loss not in ('smoothed_huber', 'pseudo_huber'):
        raise ValueError(f"loss must be 'smoothed_huber' or 'pseudo_huber', got {loss!r}")

    n_samples, n_features = x.shape
    if n_samples != y.shape[0]:
        raise ValueError(f"X has {n_samples} samples, but y has {y.shape[0]} samples.")
    if not (isinstance(tau, (int, float)) and tau >= 0):
        raise ValueError("tau must be a non-negative number.")
    if not (isinstance(delta, (int, float)) and delta > 0):
        raise ValueError("delta must be a positive number.")

    if loss == 'smoothed_huber':
        if not (isinstance(eta, (int, float)) and eta > 0):
            raise ValueError("eta must be a positive number for smoothed_huber loss.")
        if eta > delta:
            raise ValueError(f"eta ({eta}) must be <= delta ({delta}) for smoothed_huber loss.")

    if n_features == 0:
        return np.array([])
    if n_samples == 0:
        return np.zeros(n_features)

    if initial_beta is None:
        beta_initial = np.zeros(n_features)
    elif isinstance(initial_beta, np.ndarray) and initial_beta.shape == (n_features,):
        beta_initial = initial_beta
    else:
        raise ValueError(f"initial_beta must be a numpy array of shape ({n_features},) or None.")

    args = (x, y, tau, delta, eta, loss)
    optimizer_options = {'maxiter': max_iter, 'gtol': tol}
    result = minimize(
        _objective_function,
        beta_initial,
        args=args,
        jac=_gradient_function,
        method='L-BFGS-B',
        options=optimizer_options,
    )

    if not result.success:
        print(f"Warning: Optimization may not have converged. Message: {result.message}")

    return result.x
