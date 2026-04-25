import os
import sys

import numpy as np
from sklearn.linear_model import LassoCV

# 让 transrr_lib 可被 import（experiment/ 在 simu_compare/ 的上一级）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib.find_tau_opt import find_optimal_tau_robust_ridge
from transrr_lib.robust_ridge_optimizer import solve_robust_ridge
from TransLasso_functions import trans_lasso


def apply_trans_lasso(X, Y, X1, Y1):
    # 合并数据
    n = len(Y)
    nn = len(Y1)
    n_vec = [n, nn]

    X_combined = np.vstack([X, X1])
    y_combined = np.concatenate([Y, Y1])

    # 应用Trans.lasso方法
    I_til_1 = list(range(int(n_vec[0] / 4)))
    prop_re1 = trans_lasso(X_combined, y_combined, n_vec, I_til_1, l1=True)

    I_til_2 = list(range(int(3 * n_vec[0] / 4), n_vec[0]))
    prop_re2 = trans_lasso(X_combined, y_combined, n_vec, I_til_2, l1=True)

    # 结果平均
    beta_prop = (prop_re1["beta_hat"] + prop_re2["beta_hat"]) / 2

    return beta_prop

# # --- Helper Function Stubs (请用您的实际函数替换它们) ---
# def find_optimal_tau_robust_ridge(X, y, tau_range):
#     """
#     占位符函数：找到鲁棒岭回归的最优tau。
#     在实际应用中，这里应该是您实现的搜索最优tau的逻辑。
#     """
#     print(f"调用 find_optimal_tau_robust_ridge (占位符): X.shape={X.shape}, y.shape={len(y) if y is not None else 'None'}")
#     if tau_range is None or len(tau_range) == 0:
#         return 0.1 # 返回一个默认值
#     return tau_range[len(tau_range) // 2] # 简单地返回tau_range的中间值作为占位符
#
# def solve_robust_ridge(X, y, tau, delta, eta, initial_beta):
#     """
#     占位符函数：解决鲁棒岭回归问题。
#     在实际应用中，这里应该是您实现的鲁棒岭回归求解器。
#     """
#     print(f"调用 solve_robust_ridge (占位符): X.shape={X.shape}, y.shape={len(y) if y is not None else 'None'}, tau={tau}, delta={delta}, eta={eta}")
#     # 返回一个基于初始猜测的简单占位符结果
#     if initial_beta is not None:
#         return initial_beta
#     return np.zeros(X.shape[1]) # 如果没有初始猜测，返回零向量

# --- 主要的参数估计函数 ---
def estimate_regression_models_tl(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10), # 示例 tau 范围
    alphas_custom=None # 示例 alpha 范围
):
    """
    估计五种不同回归模型的系数 (betahat) 和最优正则化参数 (tau/alpha)。

    参数:
    train_X (np.ndarray): 目标任务的训练特征数据。
    train_y (np.ndarray): 目标任务的训练目标变量。
    train_X1 (np.ndarray): 源任务（辅助）的训练特征数据。
    train_y1 (np.ndarray): 源任务（辅助）的训练目标变量。
    delta_param (float): solve_robust_ridge 的 delta 参数。
    eta_param (float): solve_robust_ridge 的 eta 参数。
    tau_range (np.ndarray): 用于 find_optimal_tau_robust_ridge 的 tau 候选值范围。
    alphas_custom (np.ndarray): 用于 LassoCV 的 alpha 候选值范围。

    返回:
    dict: 一个包含每种方法估计的 'betahat' 和 'optimal_tau'/'optimal_alpha' 的字典。
          对于迁移学习方法，可能包含多个最优正则化参数。
    """

    results = {}
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]

    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) 和 train_X1 ({n_features1} features) 的特征数量必须相同。")

    # 1. Single Robust Ridge
    # print("\n--- 1. Single Robust Ridge ---")
    optimal_tau_sr = find_optimal_tau_robust_ridge(train_X, train_y, tau_range=tau_range)
    initial_guess_sr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_sr * np.eye(n_features),
        train_X.T @ train_y / len(train_y)
    )
    beta_sr = solve_robust_ridge(train_X, train_y, optimal_tau_sr, delta_param, eta_param, initial_beta=initial_guess_sr)
    results["single_robust_ridge"] = {"betahat": beta_sr, "optimal_tau": optimal_tau_sr}

    # 2. Transfer Robust Ridge
    # print("\n--- 2. Transfer Robust Ridge ---")
    optimal_tau1_tr = find_optimal_tau_robust_ridge(train_X1, train_y1, tau_range=tau_range)
    initial_guess1_tr = np.linalg.solve(
        train_X1.T @ train_X1 / len(train_y1) + optimal_tau1_tr * np.eye(n_features1),
        train_X1.T @ train_y1 / len(train_y1)
    )
    w_hat_trans = solve_robust_ridge(train_X1, train_y1, optimal_tau1_tr, delta_param, eta_param, initial_beta=initial_guess1_tr)

    Y_adjusted = train_y - (train_X @ w_hat_trans) # 注意：这里是 train_X @ w_hat_trans
    optimal_tau_tr_diff = find_optimal_tau_robust_ridge(train_X, Y_adjusted, tau_range=tau_range)
    initial_guess2_tr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_tr_diff * np.eye(n_features),
        train_X.T @ Y_adjusted / len(train_y) # Y_adjusted 是 (train_y - train_X @ w_hat_trans)
    )
    beta_tr_diff = solve_robust_ridge(train_X, Y_adjusted, optimal_tau_tr_diff, delta_param, eta_param, initial_beta=initial_guess2_tr)
    beta_tr = beta_tr_diff + w_hat_trans
    results["transfer_robust_ridge"] = {
        "betahat": beta_tr,
        "optimal_tau_source": optimal_tau1_tr,
        "optimal_tau_target_diff": optimal_tau_tr_diff,
        "w_hat_source": w_hat_trans # 也存储一下源任务的估计器
    }

    # 3. Pooled Robust Ridge
    # print("\n--- 3. Pooled Robust Ridge ---")
    XX = np.vstack((train_X, train_X1))
    YY = np.concatenate((train_y, train_y1))
    optimal_tau_pr = find_optimal_tau_robust_ridge(XX, YY, tau_range=tau_range)
    initial_guess_pr = np.linalg.solve(
        XX.T @ XX / len(YY) + optimal_tau_pr * np.eye(n_features), # 假设 XX.shape[1] == n_features
        XX.T @ YY / len(YY)
    )
    beta_pr = solve_robust_ridge(XX, YY, optimal_tau_pr, delta_param, eta_param, initial_beta=initial_guess_pr)
    results["pooled_robust_ridge"] = {"betahat": beta_pr, "optimal_tau": optimal_tau_pr}

    # print("\nFinish with Ridge Methods.")

    # 4. Single Task Lasso
    # print("\n--- 4. Single Task Lasso ---")
    lasso_sl = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, train_y)
    beta_sl = lasso_sl.coef_
    results["single_task_lasso"] = {"betahat": beta_sl, "optimal_alpha": lasso_sl.alpha_}

    # 5. Transfer Lasso
    # print("\n--- 5. Transfer Lasso ---")
    beta_tl = apply_trans_lasso(train_X, train_y, train_X1, train_y1)
    results["transfer_lasso"] = {
        "betahat": beta_tl
    }
    # print("\nFinish with Lasso Methods.")

    return results


def estimate_regression_models(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10), # 示例 tau 范围
    alphas_custom=None # 示例 alpha 范围
):
    """
    估计五种不同回归模型的系数 (betahat) 和最优正则化参数 (tau/alpha)。

    参数:
    train_X (np.ndarray): 目标任务的训练特征数据。
    train_y (np.ndarray): 目标任务的训练目标变量。
    train_X1 (np.ndarray): 源任务（辅助）的训练特征数据。
    train_y1 (np.ndarray): 源任务（辅助）的训练目标变量。
    delta_param (float): solve_robust_ridge 的 delta 参数。
    eta_param (float): solve_robust_ridge 的 eta 参数。
    tau_range (np.ndarray): 用于 find_optimal_tau_robust_ridge 的 tau 候选值范围。
    alphas_custom (np.ndarray): 用于 LassoCV 的 alpha 候选值范围。

    返回:
    dict: 一个包含每种方法估计的 'betahat' 和 'optimal_tau'/'optimal_alpha' 的字典。
          对于迁移学习方法，可能包含多个最优正则化参数。
    """

    results = {}
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]

    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) 和 train_X1 ({n_features1} features) 的特征数量必须相同。")

    # 1. Single Robust Ridge
    # print("\n--- 1. Single Robust Ridge ---")
    optimal_tau_sr = find_optimal_tau_robust_ridge(train_X, train_y, tau_range=tau_range)
    initial_guess_sr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_sr * np.eye(n_features),
        train_X.T @ train_y / len(train_y)
    )
    beta_sr = solve_robust_ridge(train_X, train_y, optimal_tau_sr, delta_param, eta_param, initial_beta=initial_guess_sr)
    results["single_robust_ridge"] = {"betahat": beta_sr, "optimal_tau": optimal_tau_sr}

    # 2. Transfer Robust Ridge
    # print("\n--- 2. Transfer Robust Ridge ---")
    optimal_tau1_tr = find_optimal_tau_robust_ridge(train_X1, train_y1, tau_range=tau_range)
    initial_guess1_tr = np.linalg.solve(
        train_X1.T @ train_X1 / len(train_y1) + optimal_tau1_tr * np.eye(n_features1),
        train_X1.T @ train_y1 / len(train_y1)
    )
    w_hat_trans = solve_robust_ridge(train_X1, train_y1, optimal_tau1_tr, delta_param, eta_param, initial_beta=initial_guess1_tr)

    Y_adjusted = train_y - (train_X @ w_hat_trans) # 注意：这里是 train_X @ w_hat_trans
    optimal_tau_tr_diff = find_optimal_tau_robust_ridge(train_X, Y_adjusted, tau_range=tau_range)
    initial_guess2_tr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_tr_diff * np.eye(n_features),
        train_X.T @ Y_adjusted / len(train_y) # Y_adjusted 是 (train_y - train_X @ w_hat_trans)
    )
    beta_tr_diff = solve_robust_ridge(train_X, Y_adjusted, optimal_tau_tr_diff, delta_param, eta_param, initial_beta=initial_guess2_tr)
    beta_tr = beta_tr_diff + w_hat_trans
    results["transfer_robust_ridge"] = {
        "betahat": beta_tr,
        "optimal_tau_source": optimal_tau1_tr,
        "optimal_tau_target_diff": optimal_tau_tr_diff,
        "w_hat_source": w_hat_trans # 也存储一下源任务的估计器
    }

    # 3. Pooled Robust Ridge
    # print("\n--- 3. Pooled Robust Ridge ---")
    XX = np.vstack((train_X, train_X1))
    YY = np.concatenate((train_y, train_y1))
    optimal_tau_pr = find_optimal_tau_robust_ridge(XX, YY, tau_range=tau_range)
    initial_guess_pr = np.linalg.solve(
        XX.T @ XX / len(YY) + optimal_tau_pr * np.eye(n_features), # 假设 XX.shape[1] == n_features
        XX.T @ YY / len(YY)
    )
    beta_pr = solve_robust_ridge(XX, YY, optimal_tau_pr, delta_param, eta_param, initial_beta=initial_guess_pr)
    results["pooled_robust_ridge"] = {"betahat": beta_pr, "optimal_tau": optimal_tau_pr}

    # print("\nFinish with Ridge Methods.")

    # 4. Single Task Lasso
    # print("\n--- 4. Single Task Lasso ---")
    lasso_sl = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, train_y)
    beta_sl = lasso_sl.coef_
    results["single_task_lasso"] = {"betahat": beta_sl, "optimal_alpha": lasso_sl.alpha_}

    # 5. Transfer Lasso
    # print("\n--- 5. Transfer Lasso ---")
    lasso_trans1 = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X1, train_y1)
    w_lasso_trans = lasso_trans1.coef_

    Y_offset_adjusted = train_y - (train_X @ w_lasso_trans) # 注意：这里是 train_X @ w_lasso_trans
    lasso_trans2 = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, Y_offset_adjusted)
    beta_trans_lasso_diff = lasso_trans2.coef_
    beta_tl = beta_trans_lasso_diff + w_lasso_trans
    results["transfer_lasso"] = {
        "betahat": beta_tl,
        "optimal_alpha_source": lasso_trans1.alpha_,
        "optimal_alpha_target_diff": lasso_trans2.alpha_,
        "w_hat_source_lasso": w_lasso_trans # 也存储一下源任务的Lasso估计器
    }
    # print("\nFinish with Lasso Methods.")

    return results


def estimate_regression_models_noLASSO(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10) # 示例 tau 范围
):
    """
    估计五种不同回归模型的系数 (betahat) 和最优正则化参数 (tau/alpha)。

    参数:
    train_X (np.ndarray): 目标任务的训练特征数据。
    train_y (np.ndarray): 目标任务的训练目标变量。
    train_X1 (np.ndarray): 源任务（辅助）的训练特征数据。
    train_y1 (np.ndarray): 源任务（辅助）的训练目标变量。
    delta_param (float): solve_robust_ridge 的 delta 参数。
    eta_param (float): solve_robust_ridge 的 eta 参数。
    tau_range (np.ndarray): 用于 find_optimal_tau_robust_ridge 的 tau 候选值范围。
    alphas_custom (np.ndarray): 用于 LassoCV 的 alpha 候选值范围。

    返回:
    dict: 一个包含每种方法估计的 'betahat' 和 'optimal_tau'/'optimal_alpha' 的字典。
          对于迁移学习方法，可能包含多个最优正则化参数。
    """

    results = {}
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]

    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) 和 train_X1 ({n_features1} features) 的特征数量必须相同。")

    # 1. Single Robust Ridge
    # print("\n--- 1. Single Robust Ridge ---")
    optimal_tau_sr = find_optimal_tau_robust_ridge(train_X, train_y, tau_range=tau_range)
    initial_guess_sr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_sr * np.eye(n_features),
        train_X.T @ train_y / len(train_y)
    )
    beta_sr = solve_robust_ridge(train_X, train_y, optimal_tau_sr, delta_param, eta_param, initial_beta=initial_guess_sr)
    results["single_robust_ridge"] = {"betahat": beta_sr, "optimal_tau": optimal_tau_sr}

    # 2. Transfer Robust Ridge
    # print("\n--- 2. Transfer Robust Ridge ---")
    optimal_tau1_tr = find_optimal_tau_robust_ridge(train_X1, train_y1, tau_range=tau_range)
    initial_guess1_tr = np.linalg.solve(
        train_X1.T @ train_X1 / len(train_y1) + optimal_tau1_tr * np.eye(n_features1),
        train_X1.T @ train_y1 / len(train_y1)
    )
    w_hat_trans = solve_robust_ridge(train_X1, train_y1, optimal_tau1_tr, delta_param, eta_param, initial_beta=initial_guess1_tr)

    Y_adjusted = train_y - (train_X @ w_hat_trans) # 注意：这里是 train_X @ w_hat_trans
    optimal_tau_tr_diff = find_optimal_tau_robust_ridge(train_X, Y_adjusted, tau_range=tau_range)
    initial_guess2_tr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_tr_diff * np.eye(n_features),
        train_X.T @ Y_adjusted / len(train_y) # Y_adjusted 是 (train_y - train_X @ w_hat_trans)
    )
    beta_tr_diff = solve_robust_ridge(train_X, Y_adjusted, optimal_tau_tr_diff, delta_param, eta_param, initial_beta=initial_guess2_tr)
    beta_tr = beta_tr_diff + w_hat_trans
    results["transfer_robust_ridge"] = {
        "betahat": beta_tr,
        "optimal_tau_source": optimal_tau1_tr,
        "optimal_tau_target_diff": optimal_tau_tr_diff,
        "w_hat_source": w_hat_trans # 也存储一下源任务的估计器
    }

    # 3. Pooled Robust Ridge
    # print("\n--- 3. Pooled Robust Ridge ---")
    XX = np.vstack((train_X, train_X1))
    YY = np.concatenate((train_y, train_y1))
    optimal_tau_pr = find_optimal_tau_robust_ridge(XX, YY, tau_range=tau_range)
    initial_guess_pr = np.linalg.solve(
        XX.T @ XX / len(YY) + optimal_tau_pr * np.eye(n_features), # 假设 XX.shape[1] == n_features
        XX.T @ YY / len(YY)
    )
    beta_pr = solve_robust_ridge(XX, YY, optimal_tau_pr, delta_param, eta_param, initial_beta=initial_guess_pr)
    results["pooled_robust_ridge"] = {"betahat": beta_pr, "optimal_tau": optimal_tau_pr}

    # print("\nFinish with Ridge Methods.")

    return results

