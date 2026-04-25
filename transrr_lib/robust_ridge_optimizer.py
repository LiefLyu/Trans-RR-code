# robust_ridge_optimizer.py

import numpy as np
from scipy.optimize import minimize


# --- 定义 rho_eta 和 psi_eta (内部辅助函数) ---
def _rho_eta_scalar(x, delta, eta):
    """Scalar version of the rho_eta loss function."""
    c_rho = -eta ** 2 / 6 + eta * delta / 2 - delta ** 2 / 2
    abs_x = np.abs(x)
    if abs_x <= delta - eta:
        val = x ** 2 / 2
    elif (delta - eta) < abs_x < delta:
        val = (delta - eta / 2) * abs_x + (delta - abs_x) ** 3 / (6 * eta) + c_rho
    else:  # abs_x >= delta
        val = (delta - eta / 2) * abs_x + c_rho
    return val


def _psi_eta_scalar(x, delta, eta):
    """Scalar version of the psi_eta function (derivative of rho_eta)."""
    abs_x = np.abs(x)
    if abs_x <= delta - eta:
        val = x
    elif (delta - eta) < abs_x < delta:
        val = np.sign(x) * (delta - eta / 2 - (delta - abs_x) ** 2 / (2 * eta))
    else:  # abs_x >= delta
        val = np.sign(x) * (delta - eta / 2)
    return val


# Vectorized versions for efficient computation
_rho_eta_vec = np.vectorize(_rho_eta_scalar, excluded=['delta', 'eta'])
_psi_eta_vec = np.vectorize(_psi_eta_scalar, excluded=['delta', 'eta'])


# --- 定义目标函数和梯度 (内部辅助函数) ---
def _objective_function(beta, x, y, tau, delta, eta):
    """Objective function to be minimized."""
    n = len(y)
    if n == 0:
        return (tau / 2) * np.sum(beta ** 2)

    residuals = y - x @ beta
    loss_sum = np.sum(_rho_eta_vec(residuals, delta=delta, eta=eta))
    regularization_term = (tau / 2) * np.sum(beta ** 2)

    return (1 / n) * loss_sum + regularization_term


def _gradient_function(beta, x, y, tau, delta, eta):
    """Gradient of the objective function."""
    n = len(y)
    if n == 0:
        return tau * beta

    residuals = y - x @ beta
    psi_values = _psi_eta_vec(residuals, delta=delta, eta=eta)
    grad_loss = - (1 / n) * x.T @ psi_values
    grad_reg = tau * beta

    return grad_loss + grad_reg


# --- 主优化函数 ---
def solve_robust_ridge(x, y, tau, delta, eta, initial_beta=None, tol=1e-6, max_iter=10000):
    """
    Solves the robust ridge regression problem:
    argmin_{beta} [ (1/n) * sum_{i=1}^{n} rho_eta(y_i - x_i^T beta) + (tau/2) * ||beta||^2 ]

    Parameters:
    ----------
    X : numpy.ndarray
        Design matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples,).
    tau : float
        L2 regularization parameter. Must be non-negative.
    delta : float
        Parameter for the rho_eta and psi_eta functions. Must be positive.
    eta : float
        Parameter for the rho_eta and psi_eta functions. Must be positive and eta <= delta.
    initial_beta : numpy.ndarray, optional
        Initial guess for beta, shape (n_features,). If None, defaults to zeros.
    tol : float, optional
        Tolerance for termination by the gradient norm for the L-BFGS-B optimizer.
    max_iter : int, optional
        Maximum number of iterations for the optimizer.

    Returns:
    -------
    numpy.ndarray or None
        The estimated coefficient vector beta_hat (shape (n_features,)).
        Returns the result from the optimizer even if not fully converged,
        check optimizer's success flag or message if precise convergence is critical.
        Returns a zero vector if n_samples is 0 but n_features > 0.
        Returns an empty array if n_features is 0.
    """
    if not (isinstance(x, np.ndarray) and x.ndim == 2):
        raise ValueError("X must be a 2D numpy array.")
    if not (isinstance(y, np.ndarray) and y.ndim == 1):
        raise ValueError("y must be a 1D numpy array.")

    n_samples, n_features = x.shape

    if n_samples != y.shape[0]:
        raise ValueError(f"X has {n_samples} samples, but y has {y.shape[0]} samples.")
    if not (isinstance(tau, (int, float)) and tau >= 0):
        raise ValueError("tau must be a non-negative number.")
    if not (isinstance(delta, (int, float)) and delta > 0):
        raise ValueError("delta must be a positive number.")
    if not (isinstance(eta, (int, float)) and eta > 0):
        raise ValueError("eta must be a positive number.")
    if eta > delta:
        raise ValueError(f"eta ({eta}) must be less than or equal to delta ({delta}).")

    if n_features == 0:
        return np.array([])
    if n_samples == 0:  # No data to estimate beta from, only regularization term affects objective if n_features > 0
        # Objective becomes (tau/2) * ||beta||^2, minimized at beta = 0
        return np.zeros(n_features)

    if initial_beta is None:
        beta_initial = np.zeros(n_features)
    elif isinstance(initial_beta, np.ndarray) and initial_beta.shape == (n_features,):
        beta_initial = initial_beta
    else:
        raise ValueError(f"initial_beta must be a numpy array of shape ({n_features},) or None.")

    args = (x, y, tau, delta, eta)

    # Note: 'disp': False by default. Caller can pass optimizer_options to override.
    optimizer_options = {'maxiter': max_iter, 'gtol': tol}

    result = minimize(_objective_function,
                      beta_initial,
                      args=args,
                      jac=_gradient_function,
                      method='L-BFGS-B',
                      options=optimizer_options)

    if not result.success:
        print(f"Warning: Optimization may not have converged. Message: {result.message}")

    return result.x