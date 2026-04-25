import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
VERIFY_DIR = THIS_DIR.parent / "simu_verify"
sys.path.insert(0, str(VERIFY_DIR))

from risk_solver import solve_cauchy, solve_gaussian_by_gap, solve_mix  # noqa: E402


DEFAULT_TAUS = [0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
DIST_ORDER = ["gaussian", "cauchy", "mix"]
DIST_LABELS = {
    "gaussian": "Gaussian",
    "cauchy": "Cauchy",
    "mix": "Mixture",
}


def solve_curve_task(task):
    distribution = task["distribution"]
    sigma = task["sigma"]
    gamma = task["gamma"]
    delta = task["delta"]
    eta = task["eta"]
    kappa = task["kappa"]
    tau = task["tau"]
    mismatch_grid = task["mismatch_grid"]

    rows = []
    initial_guess = None

    for mismatch in mismatch_grid:
        if distribution == "gaussian":
            try:
                c, r2 = solve_gaussian_by_gap(
                    sigma=sigma,
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    gap_norm=float(mismatch),
                    initial_guess=initial_guess,
                )
            except RuntimeError:
                c, r2 = solve_gaussian_by_gap(
                    sigma=sigma,
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    gap_norm=float(mismatch),
                    initial_guess=None,
                )
        elif distribution == "cauchy":
            beta_0 = np.array([float(mismatch)])
            w_hat = np.zeros(1)
            try:
                c, r2 = solve_cauchy(
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    beta_0=beta_0,
                    w_hat=w_hat,
                    gamma=gamma,
                    initial_guess=initial_guess,
                )
            except RuntimeError:
                c, r2 = solve_cauchy(
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    beta_0=beta_0,
                    w_hat=w_hat,
                    gamma=gamma,
                    initial_guess=None,
                )
        else:
            beta_0 = np.array([float(mismatch)])
            w_hat = np.zeros(1)
            try:
                c, r2 = solve_mix(
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    beta_0=beta_0,
                    w_hat=w_hat,
                    gamma=gamma,
                    sigma=sigma,
                    initial_guess=initial_guess,
                )
            except RuntimeError:
                c, r2 = solve_mix(
                    delta=delta,
                    eta=eta,
                    kappa=kappa,
                    tau=tau,
                    beta_0=beta_0,
                    w_hat=w_hat,
                    gamma=gamma,
                    sigma=sigma,
                    initial_guess=None,
                )

        r = math.sqrt(max(r2, 0.0))
        initial_guess = np.array([c, r], dtype=float)
        rows.append(
            {
                "distribution": distribution,
                "tau": float(tau),
                "mismatch": float(mismatch),
                "c": float(c),
                "r": float(r),
                "r2": float(r2),
            }
        )

    return rows


def write_csv(rows, path):
    fieldnames = ["distribution", "tau", "mismatch", "c", "r", "r2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pdf(rows, taus, colors, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), sharey=True, constrained_layout=True)

    legend_handles = []
    legend_labels = []

    for ax, distribution in zip(axes, DIST_ORDER):
        for tau, color in zip(taus, colors):
            subset = [
                row for row in rows
                if row["distribution"] == distribution and row["tau"] == float(tau)
            ]
            xs = [row["mismatch"] for row in subset]
            ys = [row["r"] for row in subset]
            line, = ax.plot(xs, ys, color=color, linewidth=2.2)
            if distribution == "gaussian":
                legend_handles.append(line)
                legend_labels.append(rf"$\tau = {tau:g}$")

        ax.set_title(DIST_LABELS[distribution], fontsize=12, pad=8)
        ax.set_xlabel(r"$\|\mathbf{\beta}_0 - \hat{\mathbf{w}}\|$", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(r"$r_\rho$", fontsize=12)
    xmax = max(row["mismatch"] for row in rows)
    ymax = max(row["r"] for row in rows)
    for ax in axes:
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(0.0, 1.05 * ymax)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(taus),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate a combined mismatch-curve figure for Gaussian, Cauchy, and mixed errors.")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.35)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--m-max", type=float, default=10.0)
    parser.add_argument("--num-points-gaussian", type=int, default=101)
    parser.add_argument("--num-points-heavy", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    args = parser.parse_args()

    taus = list(args.taus)
    colors = DEFAULT_COLORS[: len(taus)]

    tasks = []
    for tau in taus:
        tasks.append(
            {
                "distribution": "gaussian",
                "sigma": args.sigma,
                "gamma": args.gamma,
                "delta": args.delta,
                "eta": args.eta,
                "kappa": args.kappa,
                "tau": tau,
                "mismatch_grid": np.linspace(0.0, args.m_max, args.num_points_gaussian),
            }
        )
        for distribution in ["cauchy", "mix"]:
            tasks.append(
                {
                    "distribution": distribution,
                    "sigma": args.sigma,
                    "gamma": args.gamma,
                    "delta": args.delta,
                    "eta": args.eta,
                    "kappa": args.kappa,
                    "tau": tau,
                    "mismatch_grid": np.linspace(0.0, args.m_max, args.num_points_heavy),
                }
            )

    all_rows = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as executor:
        futures = [executor.submit(solve_curve_task, task) for task in tasks]
        for future in as_completed(futures):
            all_rows.extend(future.result())

    all_rows.sort(key=lambda row: (DIST_ORDER.index(row["distribution"]), row["tau"], row["mismatch"]))

    csv_path = THIS_DIR / "combined_mismatch_curves.csv"
    pdf_path = THIS_DIR / "combined_mismatch_curves.pdf"
    write_csv(all_rows, csv_path)
    write_pdf(all_rows, taus, colors, pdf_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
