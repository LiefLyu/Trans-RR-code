import scipy.integrate as spi
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import scipy.stats as stats
from numpy.linalg import norm
from numpy import linalg

#############################################################
############ Gaussian ######################################

DEFAULT_INITIAL_GUESS = np.array([1.0, 1.0])


def _solve_system(equations, initial_guess=None):
    if initial_guess is None:
        initial_guess = DEFAULT_INITIAL_GUESS
    solution, info, ier, msg = fsolve(equations, np.asarray(initial_guess, dtype=float), full_output=True)
    if ier != 1:
        raise RuntimeError(f"fsolve failed to converge: {msg}")
    return solution


def solve_gaussian_by_gap(sigma, delta, eta, kappa, tau, gap_norm, initial_guess=None):
    def equations(vars):
        c, r = vars
        sd_w = np.sqrt(sigma**2 + r**2)
        threshold1 = (delta - eta) * (c + 1)
        threshold2 = delta * (c + 1) - c * eta / 2

        normal_dist = stats.norm(0, sd_w)


        p1_1 = (2 * normal_dist.cdf(threshold1) - 1) / (1 + c)
        p1_3 = 2 * (1 - normal_dist.cdf(threshold2))
        def f1(x):
            return normal_dist.pdf(x) * 2*eta / c / np.sqrt(4*(delta + eta/c)**2 - 4*(2*eta*( np.abs(x)/c - delta + eta/2) + delta**2))
        val1, err1 = quad(f1, threshold1, threshold2)
        p1_2 = 2*val1

        eq1_l = p1_1 + p1_2 + p1_3 - 1 + kappa - tau*c

        def f2(x):
            return normal_dist.pdf(x)*x**2
        val2, err2 = quad(f2, -threshold1, threshold1)
        p2_1 = (c / (1 + c))**2 * val2
        p2_3 = (delta - eta/2)**2 * c**2 * 2 * (1 - normal_dist.cdf(threshold2))
        def f3(x):
            a = 1
            b = -np.sign(x)*2*(delta+eta/c)
            d = 2*eta*(abs(x)/c - delta + eta/2) + delta**2
            return normal_dist.pdf(x)*(x - (-b - np.sign(x)*np.sqrt(b**2 - 4*a*d)) / (2*a))**2
        val3, err3 = quad(f3, threshold1, threshold2)
        p2_2 = 2*val3
        eq2_l = kappa * (p2_1 + p2_2 + p2_3) + tau**2 * gap_norm**2 * c**2 - kappa**2 * r**2

        return [eq1_l, eq2_l]

    solution = _solve_system(equations, initial_guess=initial_guess)
    c, r = solution
    r2 = r**2

    return c, r2


def solve_gaussian(sigma, delta, eta, kappa, tau, beta_0, w_hat, initial_guess=None):
    gap_norm = np.linalg.norm(beta_0 - w_hat)
    return solve_gaussian_by_gap(sigma, delta, eta, kappa, tau, gap_norm, initial_guess=initial_guess)

#############################################################
############ Cauchy #######################################

def solve_cauchy(delta, eta, kappa, tau, beta_0, w_hat, gamma=1, initial_guess=None):
    gap_norm = np.linalg.norm(beta_0 - w_hat)

    def equations(vars):
        c, r = vars

        result1_c, result2_c = E_cauchy(delta, eta, c, r, gamma=gamma)

        eq1_l = result1_c - 1 + kappa - tau * c
        eq2_l = kappa * result2_c + tau ** 2 * gap_norm ** 2 * c ** 2 - kappa ** 2 * r ** 2

        return [eq1_l, eq2_l]

    solution = _solve_system(equations, initial_guess=initial_guess)
    c, r = solution
    r2 = r ** 2

    return c, r2


def eps_pdf(e, gamma=1):
    return gamma / (np.pi * (e**2 + gamma**2))


def eps_cdf(e, gamma=1):
    return (1.0 / np.pi) * np.arctan(e / gamma) + 0.5


def normal_pdf(z):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)


def lambda_pdf(lambd, a=0, b=np.sqrt(3)):
    return 1 / (b - a)


def d_prox_3(x, c, delta, eta):
    abs_x = np.abs(x)

    a = 1
    b = 2 * (delta + eta / c)
    d = 2 * eta * (abs_x / c - delta + eta / 2) + delta ** 2

    return 2 * eta / (c * np.sqrt(b ** 2 - 4 * d))


def prox_3(x, c, delta, eta):
    abs_x = np.abs(x)

    a = 1
    b = np.sign(x) * 2 * (delta + eta / c)
    d = 2 * eta * (abs_x / c - delta + eta / 2) + delta ** 2

    # 计算解
    sqrt_term = np.sqrt(b ** 2 - 4 * a * d)
    return (b - np.sign(x) * sqrt_term) / (2 * a)


def E_cauchy(delta, eta, c, r, gamma=1):
    def threshold1(c, lambd):
        return (delta - eta) * (c * lambd ** 2 + 1)

    def threshold2(c, lambd):
        return delta * (c * lambd ** 2 + 1) - c * lambd ** 2 * eta / 2

    ######################### eq 1 #####################################
    def integrand_11(z, lambd):
        return ((eps_cdf(threshold1(c, lambd) - r * lambd * z, gamma) - eps_cdf(-threshold1(c, lambd) - r * lambd * z, gamma))
                * normal_pdf(z) * lambda_pdf(lambd) / (1 + c * lambd ** 2))

    # 计算外层积分
    result_11, _ = spi.dblquad(
        lambda z, lambd: integrand_11(z, lambd),
        0, np.sqrt(3),
        lambda lambd: -np.inf,  # Lower limit for z
        lambda lambd: np.inf  # Upper limit for z
    )

    def integrand_12(z, lambd):
        return (1 - eps_cdf(threshold2(c, lambd) - r * lambd * z, gamma)) * normal_pdf(z) * lambda_pdf(lambd)

    result_12, _ = spi.dblquad(
        lambda z, lambd: integrand_12(z, lambd),
        0, np.sqrt(3),
        lambda lambd: -np.inf,  # Lower limit for z
        lambda lambd: np.inf  # Upper limit for z
    )
    result_12 = 2 * result_12

    def integrand_13(z, lambd):
        integral_result, _ = spi.quad(
            lambda epsilon: d_prox_3(epsilon + r * lambd * z, c * lambd ** 2, delta, eta) * eps_pdf(epsilon, gamma),
            threshold1(c, lambd) - r * lambd * z, threshold2(c, lambd) - r * lambd * z)
        return integral_result * normal_pdf(z) * lambda_pdf(lambd)

    result_13, _ = spi.dblquad(
        lambda z, lambd: integrand_13(z, lambd),
        0, np.sqrt(3),
        lambda lambd: -np.inf,  # Lower limit for z
        lambda lambd: np.inf  # Upper limit for z
    )
    result_13 = 2 * result_13

    result1 = result_11 + result_12 + result_13

    #     eq1_l = result1 - 1 + kappa - tau*c

    ###################### eq2  ######################
    def integrand_21(z, lambd):
        integral_result, _ = spi.quad(lambda epsilon: (epsilon + r * lambd * z) ** 2 * eps_pdf(epsilon, gamma),
                                      -threshold1(c, lambd) - r * lambd * z, threshold1(c, lambd) - r * lambd * z)
        return integral_result * normal_pdf(z) * lambda_pdf(lambd) * lambd ** 2 * (c / (1 + c * lambd ** 2)) ** 2

    result_21, _ = spi.dblquad(
        lambda z, lambd: integrand_21(z, lambd),
        0, np.sqrt(3),
        lambda lambd: -np.inf,  # Lower limit for z
        lambda lambd: np.inf  # Upper limit for z
    )

    def integrand_22(z, lambd):
        return ((1 - eps_cdf(threshold2(c, lambd) - r * lambd * z, gamma)) * (delta - eta / 2) ** 2 * c ** 2 * lambd ** 2
                * normal_pdf(z) * lambda_pdf(lambd))

    result_22, _ = spi.dblquad(
        lambda z, lambd: integrand_22(z, lambd),
        0, np.sqrt(3),
        lambda lambd: -np.inf,  # Lower limit for z
        lambda lambd: np.inf  # Upper limit for z
    )
    result_22 = 2 * result_22

    result_23, _ = spi.tplquad(lambda epsilon, z, lambd: (
                (epsilon + r * lambd * z - prox_3(epsilon + r * lambd * z, c * lambd ** 2, delta, eta)) ** 2
                * eps_pdf(epsilon, gamma) * normal_pdf(z) * lambda_pdf(lambd) / lambd ** 2),  # 函数
                               0,  # lambd下界
                               np.sqrt(3),  # lambd上界
                               lambda lambd: -np.inf,  # z下界
                               lambda lambd: np.inf,  # z上界
                               lambda lambd, z: threshold1(c, lambd) - r * lambd * z,  # epsilon下界
                               lambda lambd, z: threshold2(c, lambd) - r * lambd * z)  # epsilon上界

    result_23 = 2 * result_23

    result2 = result_21 + result_22 + result_23

    #     eq2_l = kappa * result2 + tau**2 * norm_0 * c**2 - kappa**2 * r**2

    return result1, result2

################################################################
################## mix ########################################

def solve_mix(delta, eta, kappa, tau, beta_0, w_hat, gamma=1, sigma=1, initial_guess=None):
    gap_norm = np.linalg.norm(beta_0 - w_hat)

    def equations(vars):
        c, r = vars

        result1_c, result2_c = E_cauchy(delta, eta, c, r, gamma=gamma)
        result1_g, result2_g = E_gaussian(delta, eta, c, r, sigma=sigma)

        eq1_l = (result1_c + result1_g) / 2 - 1 + kappa - tau * c
        eq2_l = kappa * (result2_c + result2_g) / 2 + tau ** 2 * gap_norm * gap_norm * c ** 2 - kappa ** 2 * r ** 2

        return [eq1_l, eq2_l]

    solution = _solve_system(equations, initial_guess=initial_guess)

    c, r = solution
    r2 = r ** 2

    return c, r2


def E_gaussian(delta, eta, c, r, sigma=1):

    sd_w = np.sqrt(sigma**2 + r**2)
    threshold1 = (delta - eta) * (c + 1)
    threshold2 = delta * (c + 1) - c * eta / 2

    normal_dist = stats.norm(0, sd_w)

    p1_1 = (2 * normal_dist.cdf(threshold1) - 1) / (1 + c)
    p1_3 = 2 * (1 - normal_dist.cdf(threshold2))
    def f1(x):
        return normal_dist.pdf(x) * 2*eta / c / np.sqrt(4*(delta + eta/c)**2 - 4*(2*eta*( np.abs(x)/c - delta + eta/2) + delta**2))
    val1, err1 = quad(f1, threshold1, threshold2)
    p1_2 = 2*val1

#     eq1_l = p1_1 + p1_2 + p1_3 - 1 + kappa - tau*c
    result_1 = p1_1 + p1_2 + p1_3

    def f2(x):
        return normal_dist.pdf(x)*x**2
    val2, err2 = quad(f2, -threshold1, threshold1)
    p2_1 = (c / (1 + c))**2 * val2
    p2_3 = (delta - eta/2)**2 * c**2 * 2 * (1 - normal_dist.cdf(threshold2))
    def f3(x):
        a = 1
        b = -np.sign(x)*2*(delta+eta/c)
        d = 2*eta*(abs(x)/c - delta + eta/2) + delta**2
        return normal_dist.pdf(x)*(x - (-b - np.sign(x)*np.sqrt(b**2 - 4*a*d)) / (2*a))**2
    val3, err3 = quad(f3, threshold1, threshold2)
    p2_2 = 2*val3
#     eq2_l = kappa * (p2_1 + p2_2 + p2_3) + tau**2 * np.linalg.norm(delta_0 - w_hat + w_0)**2 * c**2 - kappa**2 * r**2
    result_2 = p2_1 + p2_2 + p2_3

    return result_1, result_2
