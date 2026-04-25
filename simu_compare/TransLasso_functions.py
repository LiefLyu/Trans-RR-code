import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error


def ind_set(n_vec, k_vec):
    """对应R中的ind.set函数，返回索引集合"""
    ind_re = []
    for k in k_vec:
        if k == 1:
            ind_re.extend(range(n_vec[0]))
        else:
            start = sum(n_vec[:k - 1])
            end = sum(n_vec[:k])
            ind_re.extend(range(start, end))
    return ind_re


def las_kA(X, y, A0, n_vec, l1=True, lam_const=None):
    """对应R中的las.kA函数"""
    p = X.shape[1]
    ind_1 = ind_set(n_vec, [1])
    X1 = X[ind_1]
    y1 = y[ind_1]

    if A0 is not None:
        ind_A = []
        for k in A0:
            ind_A.extend(ind_set(n_vec, [k + 1]))
        XA = X[ind_A]
        yA = y[ind_A]
        X_train = np.vstack([X1, XA])
        y_train = np.concatenate([y1, yA])

        if l1:
            if lam_const is None:
                cv_model = LassoCV(cv=5, random_state=0)
                cv_model.fit(X_train, y_train)
                lam_min = cv_model.alpha_
                lam_const = lam_min / np.sqrt(2 * np.log(p) / n_vec[0])
            else:
                lam_min = lam_const * np.sqrt(2 * np.log(p) / n_vec[0])

            model = Lasso(alpha=lam_min)
            model.fit(X_train, y_train)
            beta_kA = model.coef_

            # 计算权重向量
            w_kA = np.zeros((len(A0), p))
            for i, k in enumerate(A0):
                ind_k = ind_set(n_vec, [k + 1])
                Xk = X[ind_k]
                yk = y[ind_k]
                alpha_k = lam_const * np.sqrt(2 * np.log(p) / len(ind_k))
                model_k = Lasso(alpha=alpha_k)
                model_k.fit(Xk, yk)
                w_kA[i] = model_k.coef_
            w_kA = np.mean(w_kA, axis=0)
        else:
            # 非L1正则化实现略
            pass
    else:
        # A0是None的情况
        if l1:
            cv_model = LassoCV(cv=5, random_state=0)
            cv_model.fit(X1, y1)
            lam_min = cv_model.alpha_
            lam_const = lam_min / np.sqrt(2 * np.log(p) / n_vec[0])
            beta_kA = cv_model.coef_
            w_kA = None
        else:
            # 非L1正则化实现略
            pass

    return {"beta_kA": beta_kA, "w_kA": w_kA, "lam_const": lam_const}


def agg_fun(B, X_test, y_test):
    """对应R中的agg.fun函数，聚合估计器"""
    n_test = X_test.shape[0]
    K = B.shape[1]

    # 计算预测误差
    err_vec = np.zeros(K)
    for k in range(K):
        y_pred = X_test @ B[:, k]
        err_vec[k] = np.mean((y_test - y_pred) ** 2)

    # 计算权重
    theta = np.exp(-err_vec) / np.sum(np.exp(-err_vec))

    # 加权平均
    beta = B @ theta

    return {"beta": beta, "theta": theta}


def trans_lasso(X, y, n_vec, I_til, l1=True):
    """Python版本的Trans.lasso函数"""
    M = len(n_vec) - 1

    # 步骤1：划分数据
    X0_til = X[I_til]
    y0_til = y[I_til]
    X = np.delete(X, I_til, axis=0)
    y = np.delete(y, I_til)

    # 步骤2：计算相似性度量
    Rhat = np.zeros(M + 1)
    p = X.shape[1]
    n_vec[0] = n_vec[0] - len(I_til)
    ind_1 = ind_set(n_vec, [1])

    for k in range(2, M + 2):
        ind_k = ind_set(n_vec, [k])
        Xty_k = X[ind_k].T @ y[ind_k] / n_vec[k - 1] - X[ind_1].T @ y[ind_1] / n_vec[0]
        margin_T = np.sort(np.abs(Xty_k))[::-1][:round(n_vec[0] / 3)]
        Rhat[k - 1] = np.sum(margin_T ** 2)

    # 构建任务集合
    Tset = []
    kk_list = np.unique(np.argsort(Rhat[1:]) + 1)

    for kk in kk_list:
        Tset.append(np.where(np.argsort(Rhat[1:]) + 1 <= kk)[0])

    Tset = [tuple(sorted(t)) for t in Tset]
    Tset = list(set(Tset))

    # 估计并聚合
    beta_T = []
    init_re = las_kA(X=X, y=y, A0=None, n_vec=n_vec, l1=l1)
    beta_T.append(init_re["beta_kA"])
    beta_pool_T = [init_re["beta_kA"]]

    for T_k in Tset:
        re_k = las_kA(X=X, y=y, A0=T_k, n_vec=n_vec, l1=l1, lam_const=init_re["lam_const"])
        beta_T.append(re_k["beta_kA"])
        beta_pool_T.append(re_k["w_kA"] if re_k["w_kA"] is not None else re_k["beta_kA"])

    # 去重并转为矩阵
    beta_T = np.column_stack([b for b in beta_T])
    beta_pool_T = np.column_stack([b for b in beta_pool_T if b is not None])

    # 聚合
    agg_re1 = agg_fun(B=beta_T, X_test=X0_til, y_test=y0_til)
    agg_re2 = agg_fun(B=beta_pool_T, X_test=X0_til, y_test=y0_til)

    return {
        "beta_hat": agg_re1["beta"],
        "theta_hat": agg_re1["theta"],
        "rank_pi": np.argsort(Rhat[1:]) + 1,
        "beta_pool": agg_re2["beta"],
        "theta_pool": agg_re2["theta"]
    }