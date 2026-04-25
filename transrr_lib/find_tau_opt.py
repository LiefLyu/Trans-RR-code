import numpy as np
from sklearn.model_selection import KFold
from .robust_ridge_optimizer import solve_robust_ridge


def find_optimal_tau_robust_ridge(X, Y, n_folds=5, tau_range=None, psi_delta=1.35, psi_eta=0.1, criterion='mae'):
    """
    为 Robust Ridge 回归寻找最优的 tau 值（5 折交叉验证）。

    Args:
        X: 特征矩阵
        Y: 目标向量
        n_folds: 交叉验证 fold 数量
        tau_range: 候选 tau 值序列，None 时使用默认对数网格
        psi_delta (float): psi 函数参数 delta
        psi_eta (float): psi 函数参数 eta
        criterion (str): 验证准则，'mae'（默认）或 'mse'

    Returns:
        最优的 tau 值
    """
    if criterion not in ('mae', 'mse'):
        raise ValueError(f"criterion must be 'mae' or 'mse', got {criterion!r}")

    if tau_range is None:
        tau_range = np.logspace(-2, 1, 19, base=3)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    cv_errors = np.zeros(len(tau_range))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        for i, tau in enumerate(tau_range):
            try:
                initial_guess = np.linalg.solve(
                    X_train.T @ X_train / len(Y_train) + tau * np.eye(X_train.shape[1]),
                    X_train.T @ Y_train / len(Y_train),
                )
                delta_hat = solve_robust_ridge(X_train, Y_train, tau, psi_delta, psi_eta, initial_beta=initial_guess)

                residuals = Y_val - X_val @ delta_hat
                if criterion == 'mae':
                    val_error = np.mean(np.abs(residuals))
                else:
                    val_error = np.mean(residuals ** 2)
                cv_errors[i] += val_error / n_folds
            except Exception as e:
                cv_errors[i] += np.inf
                print(f"Warning: Optimization failed for tau={tau} in fold {fold_idx}. Error: {e}")

    best_tau_idx = np.argmin(cv_errors)
    return tau_range[best_tau_idx]

