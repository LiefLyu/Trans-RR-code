import os
import sys

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

# 让 transrr_lib 可被 import（experiment/ 在 simu_compare/ 的上一级）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib.find_tau_opt import find_optimal_tau_robust_ridge
from transrr_lib.robust_ridge_optimizer import solve_robust_ridge
from transrr_lib.adaptive import aggregate_by_cv
from TransLasso_functions import trans_lasso

# Adaptive uses an independent K-fold partition for theta selection,
# distinct from the partition used inside find_optimal_tau_robust_ridge
# (which is KFold(random_state=1)).
_ADAPTIVE_KFOLD_SEED = 2


# ============================================================
# Trans-Lasso baseline (uses TransLasso_functions)
# ============================================================
def apply_trans_lasso(X, Y, X1, Y1):
    n = len(Y)
    nn = len(Y1)
    n_vec = [n, nn]
    X_combined = np.vstack([X, X1])
    y_combined = np.concatenate([Y, Y1])

    I_til_1 = list(range(int(n_vec[0] / 4)))
    prop_re1 = trans_lasso(X_combined, y_combined, n_vec, I_til_1, l1=True)

    I_til_2 = list(range(int(3 * n_vec[0] / 4), n_vec[0]))
    prop_re2 = trans_lasso(X_combined, y_combined, n_vec, I_til_2, l1=True)

    return (prop_re1["beta_hat"] + prop_re2["beta_hat"]) / 2


# ============================================================
# Shared ridge block: Single-RR, Trans-RR, Pooled-RR, Adaptive
#
# This is the ridge half of the comparison pipeline. Used by all
# three estimate_regression_models_* variants.
# ============================================================
def _compute_ridge_methods_with_adaptive(
    train_X, train_y,
    train_X1, train_y1,
    delta_param, eta_param,
    tau_range,
    criterion='mae',
    loss='smoothed_huber',
):
    """
    Fits Single-RR, Trans-RR, Pooled-RR plus the Adaptive aggregate
    beta_ada = theta * beta_trans + (1 - theta) * beta_single.

    Tau-tuning for Single-RR and Trans-RR target step uses
    KFold(seed=1) inside find_optimal_tau_robust_ridge.
    The Adaptive layer then draws an INDEPENDENT KFold(seed=2)
    partition of the target sample, refits the K fold-level
    base estimators at the already-tuned tau and tau_st (with
    w_hat held fixed), and picks theta on a grid by minimising
    the convex-combination CV loss on this independent partition
    (Algorithm 2, Steps 1--4 in the paper).

    Returns
    -------
    dict with keys 'single_robust_ridge', 'transfer_robust_ridge',
    'pooled_robust_ridge', 'adaptive'.
    """
    n_features = train_X.shape[1]

    # 1. Single-RR (per-fold info kept for Adaptive)
    res_sr = find_optimal_tau_robust_ridge(
        train_X, train_y, tau_range=tau_range,
        psi_delta=delta_param, psi_eta=eta_param,
        criterion=criterion, loss=loss,
        return_folds=True,
    )
    optimal_tau_sr = res_sr['tau_opt']
    initial_guess_sr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_sr * np.eye(n_features),
        train_X.T @ train_y / len(train_y),
    )
    beta_sr = solve_robust_ridge(
        train_X, train_y, optimal_tau_sr, delta_param, eta_param,
        initial_beta=initial_guess_sr, loss=loss,
    )

    # 2. Trans-RR
    # 2a. Source step (its own CV; fold info not needed)
    optimal_tau1_tr = find_optimal_tau_robust_ridge(
        train_X1, train_y1, tau_range=tau_range,
        psi_delta=delta_param, psi_eta=eta_param,
        criterion=criterion, loss=loss,
    )
    initial_guess1_tr = np.linalg.solve(
        train_X1.T @ train_X1 / len(train_y1) + optimal_tau1_tr * np.eye(n_features),
        train_X1.T @ train_y1 / len(train_y1),
    )
    w_hat_trans = solve_robust_ridge(
        train_X1, train_y1, optimal_tau1_tr, delta_param, eta_param,
        initial_beta=initial_guess1_tr, loss=loss,
    )

    # 2b. Target residual step (per-fold info kept for Adaptive)
    Y_adjusted = train_y - (train_X @ w_hat_trans)
    res_tr_diff = find_optimal_tau_robust_ridge(
        train_X, Y_adjusted, tau_range=tau_range,
        psi_delta=delta_param, psi_eta=eta_param,
        criterion=criterion, loss=loss,
        return_folds=True,
    )
    optimal_tau_tr_diff = res_tr_diff['tau_opt']
    initial_guess2_tr = np.linalg.solve(
        train_X.T @ train_X / len(train_y) + optimal_tau_tr_diff * np.eye(n_features),
        train_X.T @ Y_adjusted / len(train_y),
    )
    beta_tr_diff = solve_robust_ridge(
        train_X, Y_adjusted, optimal_tau_tr_diff, delta_param, eta_param,
        initial_beta=initial_guess2_tr, loss=loss,
    )
    beta_tr = beta_tr_diff + w_hat_trans

    # 3. Pooled-RR
    XX = np.vstack((train_X, train_X1))
    YY = np.concatenate((train_y, train_y1))
    optimal_tau_pr = find_optimal_tau_robust_ridge(
        XX, YY, tau_range=tau_range,
        psi_delta=delta_param, psi_eta=eta_param,
        criterion=criterion, loss=loss,
    )
    initial_guess_pr = np.linalg.solve(
        XX.T @ XX / len(YY) + optimal_tau_pr * np.eye(n_features),
        XX.T @ YY / len(YY),
    )
    beta_pr = solve_robust_ridge(
        XX, YY, optimal_tau_pr, delta_param, eta_param,
        initial_beta=initial_guess_pr, loss=loss,
    )

    # 4. Adaptive aggregation on an INDEPENDENT K-fold partition.
    # The base estimators' tau-tuning used KFold(random_state=1) inside
    # find_optimal_tau_robust_ridge. To keep theta selection from being
    # contaminated by the same partition that picked the taus, we draw a
    # separate K-fold partition (seed=2) and refit the K fold-level
    # base estimators at the already-tuned (tau, tau_st) and the fixed w_hat.
    # This adds 2*K extra solver calls per simulation replicate
    # (about 5% of the overall computation).
    kf_ada = KFold(n_splits=5, shuffle=True, random_state=_ADAPTIVE_KFOLD_SEED)
    fold_indices = list(kf_ada.split(train_X))

    fold_betas_single = []
    fold_betas_trans = []
    for train_idx, _ in fold_indices:
        X_tr = train_X[train_idx]
        Y_tr = train_y[train_idx]
        n_tr = len(Y_tr)

        # Fold-level Single-RR at tuned tau_st = optimal_tau_sr
        init_sr = np.linalg.solve(
            X_tr.T @ X_tr / n_tr + optimal_tau_sr * np.eye(n_features),
            X_tr.T @ Y_tr / n_tr,
        )
        beta_sr_fold = solve_robust_ridge(
            X_tr, Y_tr, optimal_tau_sr, delta_param, eta_param,
            initial_beta=init_sr, loss=loss,
        )
        fold_betas_single.append(beta_sr_fold)

        # Fold-level Trans-RR target step at tuned tau = optimal_tau_tr_diff,
        # with w_hat_trans (from full-source fit) held fixed.
        Y_adj_tr = Y_tr - X_tr @ w_hat_trans
        init_tr = np.linalg.solve(
            X_tr.T @ X_tr / n_tr + optimal_tau_tr_diff * np.eye(n_features),
            X_tr.T @ Y_adj_tr / n_tr,
        )
        delta_fold = solve_robust_ridge(
            X_tr, Y_adj_tr, optimal_tau_tr_diff, delta_param, eta_param,
            initial_beta=init_tr, loss=loss,
        )
        fold_betas_trans.append(delta_fold + w_hat_trans)

    ada_res = aggregate_by_cv(
        fold_indices=fold_indices,
        fold_betas_trans=fold_betas_trans,
        fold_betas_single=fold_betas_single,
        X=train_X, Y=train_y,
        beta_trans_full=beta_tr,
        beta_single_full=beta_sr,
        criterion=criterion,
    )

    return {
        "single_robust_ridge": {
            "betahat": beta_sr,
            "optimal_tau": optimal_tau_sr,
            "cv_loss_min": res_sr['cv_loss_min'],
        },
        "transfer_robust_ridge": {
            "betahat": beta_tr,
            "optimal_tau_source": optimal_tau1_tr,
            "optimal_tau_target_diff": optimal_tau_tr_diff,
            "w_hat_source": w_hat_trans,
            "cv_loss_min": res_tr_diff['cv_loss_min'],
        },
        "pooled_robust_ridge": {
            "betahat": beta_pr,
            "optimal_tau": optimal_tau_pr,
        },
        "adaptive": {
            "betahat": ada_res['beta_ada'],
            "theta_star": ada_res['theta_star'],
            "cv_curve": ada_res['cv_curve'].tolist(),
            "theta_grid": ada_res['theta_grid'].tolist(),
            "cv_loss_min": ada_res['cv_loss_min'],
        },
    }


# ============================================================
# Public pipelines (preserve original keys; add 'adaptive' on top)
# ============================================================
def estimate_regression_models_tl(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10),
    alphas_custom=None,
    criterion='mae',
    loss='smoothed_huber',
):
    """Single-RR / Trans-RR / Pooled-RR / Adaptive  +  Single-Lasso + Trans-Lasso (apply_trans_lasso)."""
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]
    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) and train_X1 ({n_features1} features) must match.")

    results = _compute_ridge_methods_with_adaptive(
        train_X, train_y, train_X1, train_y1,
        delta_param, eta_param, tau_range,
        criterion=criterion, loss=loss,
    )

    # Single-Task Lasso
    lasso_sl = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, train_y)
    results["single_task_lasso"] = {"betahat": lasso_sl.coef_, "optimal_alpha": lasso_sl.alpha_}

    # Trans-Lasso (Li/Cai/Li flavour via apply_trans_lasso)
    results["transfer_lasso"] = {"betahat": apply_trans_lasso(train_X, train_y, train_X1, train_y1)}

    return results


def estimate_regression_models(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10),
    alphas_custom=None,
    criterion='mae',
    loss='smoothed_huber',
):
    """Single-RR / Trans-RR / Pooled-RR / Adaptive  +  Single-Lasso + two-stage Trans-Lasso (LassoCV-based)."""
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]
    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) and train_X1 ({n_features1} features) must match.")

    results = _compute_ridge_methods_with_adaptive(
        train_X, train_y, train_X1, train_y1,
        delta_param, eta_param, tau_range,
        criterion=criterion, loss=loss,
    )

    # Single-Task Lasso
    lasso_sl = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, train_y)
    results["single_task_lasso"] = {"betahat": lasso_sl.coef_, "optimal_alpha": lasso_sl.alpha_}

    # Two-stage Trans-Lasso (source LassoCV then offset-adjusted target LassoCV)
    lasso_trans1 = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X1, train_y1)
    w_lasso_trans = lasso_trans1.coef_
    Y_offset_adjusted = train_y - (train_X @ w_lasso_trans)
    lasso_trans2 = LassoCV(fit_intercept=False, max_iter=10000, tol=1e-3, alphas=alphas_custom, cv=5, selection="random").fit(train_X, Y_offset_adjusted)
    beta_tl = lasso_trans2.coef_ + w_lasso_trans
    results["transfer_lasso"] = {
        "betahat": beta_tl,
        "optimal_alpha_source": lasso_trans1.alpha_,
        "optimal_alpha_target_diff": lasso_trans2.alpha_,
        "w_hat_source_lasso": w_lasso_trans,
    }

    return results


def estimate_regression_models_noLASSO(
    train_X, train_y,
    train_X1, train_y1,
    delta_param=1.35,
    eta_param=0.1,
    tau_range=np.logspace(-3, 1, 10),
    criterion='mae',
    loss='smoothed_huber',
):
    """Single-RR / Trans-RR / Pooled-RR / Adaptive only (no Lasso)."""
    n_features = train_X.shape[1]
    n_features1 = train_X1.shape[1]
    if n_features != n_features1:
        raise ValueError(f"train_X ({n_features} features) and train_X1 ({n_features1} features) must match.")

    return _compute_ridge_methods_with_adaptive(
        train_X, train_y, train_X1, train_y1,
        delta_param, eta_param, tau_range,
        criterion=criterion, loss=loss,
    )
