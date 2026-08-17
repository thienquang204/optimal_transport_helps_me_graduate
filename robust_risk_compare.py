#!/usr/bin/env python3
"""Compare direct and augmented-MMOT Greenkhorn solvers for robust risk.

The discrete expected-shortfall problem is the MMPOT problem

    max <payoff, X>  s.t. X >= 0, r_k(X) <= p_k, sum(X) = s.

``GreenkhornMMPOT`` works with the slack variables directly.  The comparison
solver uses the common dummy-point reduction followed by multimarginal
Greenkhorn (Lin--Ho--Cuturi--Jordan, JMLR 2022), then extracts its all-real
block.  Importantly, feasibility of that extracted block is checked rather
than inferred from feasibility of the augmented MMOT tensor.

Only NumPy is required.  The default three 50-point marginals use about a few
MB and are suitable for a quick command-line experiment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass
class Result:
    solver: str
    n: int
    marginals: int
    mass: float
    eta: float
    tolerance: float
    iterations: int
    seconds: float
    converged: bool
    robust_risk: float
    transport_objective: float
    extracted_mass: float
    mass_violation: float
    marginal_excess: float
    partial_infeasibility: float
    max_constraint_residual: float
    augmented_infeasibility: float


def marginal(x: Array, axis: int, axis_tuple: tuple[int, ...] | None = None) -> Array:
    """Return one tensor marginal using precomputed axis tuples when available."""
    if axis_tuple is None:
        axis_tuple = tuple(i for i in range(x.ndim) if i != axis)
    return x.sum(axis=axis_tuple)


def rho(a: Array, b: Array) -> float:
    """Generalized KL divergence rho(a, b) = sum(b - a + a * log(a / b)) with exact zero handling."""
    b_safe = np.maximum(b, 1e-300)
    val = float(np.sum(b_safe - a))
    pos = a > 0
    if np.any(pos):
        val += float(np.sum(a[pos] * np.log(a[pos] / b_safe[pos])))
    return max(val, 0.0)


def fast_marginal(x: Array, k: int, m: int, axis_tuples: Sequence[tuple[int, ...]]) -> Array:
    """SIMD-accelerated contiguous tensor marginal contraction along axis k."""
    if m == 3:
        if k == 0:
            return x.reshape(x.shape[0], -1).sum(axis=1)
        elif k == 2:
            return x.reshape(-1, x.shape[2]).sum(axis=0)
        else:
            return x.sum(axis=axis_tuples[1])
    elif m == 2:
        return x.sum(axis=1) if k == 0 else x.sum(axis=0)
    else:
        return x.sum(axis=axis_tuples[k])


def partial_diagnostics(x: Array, p: Sequence[Array], s: float) -> tuple[float, ...]:
    m = x.ndim
    axis_tuples = [tuple(i for i in range(m) if i != k) for k in range(m)]
    mass = float(x.sum())
    mass_error = abs(mass - s)
    excess = sum(float(np.maximum(fast_marginal(x, k, m, axis_tuples) - p[k], 0.0).sum())
                 for k in range(m))
    residual = max(
        [mass_error]
        + [float(np.maximum(fast_marginal(x, k, m, axis_tuples) - p[k], 0.0).sum()) for k in range(m)]
    )
    return mass, mass_error, excess, mass_error + excess, residual


def _safe_kernel(cost: Array, eta: float) -> Array:
    """Compute stable entropic Gibbs kernel exp(-cost/eta) without overflow or exact zero underflow."""
    c_min = float(cost.min())
    z = -(cost - c_min) / max(float(eta), 1e-12)
    k = np.exp(np.clip(z, -600.0, 0.0))
    return np.maximum(k, np.finfo(np.float64).tiny)


def greenkhorn_mmpot(cost: Array, p: Sequence[Array], s: float, eta: float,
                     tol: float, max_iter: int) -> tuple[Array, int, bool, float]:
    """Highly optimized Algorithm 1 Greenkhorn solver for entropic multimarginal partial OT.

    Features:
    - Fast contiguous memory tensor marginal contractions
    - Precomputed contraction axes and broadcast shapes
    - Instant scalar mass projection caching
    - In-place tensor rescaling to minimize memory allocations
    - Numerically stabilized Bregman divergence & ratio updates for small eta
    """
    m = cost.ndim
    axis_tuples = [tuple(j for j in range(m) if j != k) for k in range(m)]
    shapes = [tuple(p[k].size if i == k else 1 for i in range(m)) for k in range(m)]

    x = _safe_kernel(cost, eta)
    init_mass = float(x.sum())
    x *= s / max(init_mass, 1e-300)

    # Compute initial marginals using fast SIMD contraction
    rs = [fast_marginal(x, k, m, axis_tuples) for k in range(m)]
    current_mass = float(rs[0].sum())

    # Positive slack initialization: q_k = max(p_k - r_k, eps)
    q = [np.maximum(p[k] - rs[k], 1e-300) for k in range(m)]

    start = time.perf_counter()
    converged = False

    for it in range(max_iter + 1):
        # Vectorized constraint residuals
        vector_res = [float(np.abs(rs[k] + q[k] - p[k]).sum()) for k in range(m)]
        mass_res = abs(current_mass - s)

        if max(vector_res + [mass_res]) <= tol:
            converged = True
            break
        if it == max_iter:
            break

        # Compute Bregman KL scores to greedily pick worst-violated constraint
        scores = [rho(p[k], rs[k] + q[k]) for k in range(m)]
        mass_score = rho(np.array([s], dtype=np.float64), np.array([current_mass], dtype=np.float64))
        k = int(np.argmax(scores))

        if scores[k] >= mass_score:
            denom = np.maximum(rs[k] + q[k], 1e-300)
            ratio = np.clip(p[k] / denom, 1e-15, 1e15)

            # In-place coordinate rescaling
            x *= ratio.reshape(shapes[k])
            q[k] *= ratio
            rs[k] *= ratio
            current_mass = float(rs[k].sum())

            # Only recompute remaining m - 1 marginals
            for j in range(m):
                if j != k:
                    rs[j] = fast_marginal(x, j, m, axis_tuples)
        else:
            # Mass scaling step: all marginals and mass scale by exact scalar
            scale = s / max(current_mass, 1e-300)
            x *= scale
            for j in range(m):
                rs[j] *= scale
            current_mass = s

    return x, it, converged, time.perf_counter() - start


def dummy_extension(cost: Array, p: Sequence[Array], s: float,
                    penalty: float) -> tuple[Array, list[Array]]:
    """Build the (n+1)^m finite-penalty dummy-point MMOT extension.

    The intended support consists of the all-real block and cells with exactly
    one real coordinate. Other dummy patterns get ``penalty``. With a finite
    entropic kernel they still receive positive mass, which is precisely why
    extraction can be infeasible.
    """
    m, n = cost.ndim, cost.shape[0]
    masses = np.asarray([pk.sum() for pk in p], dtype=float)
    total = float(masses.sum() - (m - 1) * s)
    ext_p = [np.r_[pk, total - masses[k]] for k, pk in enumerate(p)]
    ext_c = np.full((n + 1,) * m, float(penalty), dtype=np.float64)
    ext_c[(slice(0, n),) * m] = cost
    for k in range(m):
        idx = [n] * m
        idx[k] = slice(0, n)
        ext_c[tuple(idx)] = 0.0
    return ext_c, ext_p


def greenkhorn_mmot(cost: Array, p: Sequence[Array], eta: float, tol: float,
                    max_iter: int) -> tuple[Array, int, bool, float, float]:
    """Optimized greedy multimarginal Sinkhorn block updates of Lin--Ho et al.

    Features:
    - SIMD-accelerated contiguous tensor marginal contractions
    - Incremental score & residual tracking without redundant tensor scans
    - Precomputed broadcast shapes and memory-efficient in-place scaling
    - Numerical safeguarding for small regularization eta
    """
    m = cost.ndim
    axis_tuples = [tuple(j for j in range(m) if j != k) for k in range(m)]
    shapes = [tuple(p[k].size if i == k else 1 for i in range(m)) for k in range(m)]

    x = _safe_kernel(cost, eta)
    p0_mass = float(p[0].sum())
    x *= p0_mass / max(float(x.sum()), 1e-300)

    # Initial marginals, residuals, and Bregman divergence scores
    rs = [fast_marginal(x, k, m, axis_tuples) for k in range(m)]
    residuals = [float(np.abs(rs[k] - p[k]).sum()) for k in range(m)]
    scores = [rho(p[k], rs[k]) for k in range(m)]

    start = time.perf_counter()
    converged = False
    aug_res = np.inf

    for it in range(max_iter + 1):
        aug_res = max(residuals)
        if aug_res <= tol:
            converged = True
            break
        if it == max_iter:
            break

        # Pick worst-violated marginal
        k = int(np.argmax(scores))

        ratio = np.clip(p[k] / np.maximum(rs[k], 1e-300), 1e-15, 1e15)
        x *= ratio.reshape(shapes[k])
        rs[k] *= ratio

        # Marginal k is now updated
        residuals[k] = float(np.abs(rs[k] - p[k]).sum())
        scores[k] = rho(p[k], rs[k])

        # Recompute only the m - 1 other marginals and their corresponding scores
        for j in range(m):
            if j != k:
                rs[j] = fast_marginal(x, j, m, axis_tuples)
                residuals[j] = float(np.abs(rs[j] - p[j]).sum())
                scores[j] = rho(p[j], rs[j])

    return x, it, converged, time.perf_counter() - start, aug_res


def synthetic_problem(n: int, m: int, seed: int) -> tuple[list[Array], Array, Array]:
    """Smooth nonlinear portfolio loss and strictly positive empirical marginals."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-2.5, 2.5, n)
    marginals = []
    for k in range(m):
        loc, scale = rng.uniform(-0.45, 0.45), rng.uniform(0.65, 1.15)
        pk = np.exp(-0.5 * ((grid - loc) / scale) ** 2)
        pk *= rng.lognormal(0.0, 0.12, n)
        pk += 1e-12
        marginals.append(pk / pk.sum())

    meshes = np.meshgrid(*([grid] * m), indexing="ij", sparse=True)
    portfolio = sum((1.0 + 0.15 * k) * meshes[k] for k in range(m)) / m
    interaction = np.ones((n,) * m)
    for z in meshes:
        interaction *= np.tanh(z)
    loss = np.maximum(portfolio, 0.0) ** 2 + 0.20 * np.abs(interaction)
    payoff = (loss - loss.min()) / max(float(loss.max() - loss.min()), 1e-15)
    cost = 1.0 - payoff  # fixed mass: min cost == max payoff
    return marginals, payoff, cost


def make_result(name: str, x: Array, p: Sequence[Array], payoff: Array, cost: Array,
                s: float, eta: float, tol: float, it: int, elapsed: float,
                converged: bool, aug_inf: float = 0.0) -> Result:
    mass, mass_err, excess, infeas, residual = partial_diagnostics(x, p, s)
    gain = float(np.sum(payoff * x))
    return Result(name, cost.shape[0], cost.ndim, s, eta, tol, it, elapsed,
                  converged, gain / s, float(np.sum(cost * x)), mass, mass_err,
                  excess, infeas, residual, aug_inf)


def print_table(rows: Sequence[Result]) -> None:
    fields = ["solver", "mass", "eta", "iterations", "seconds", "robust_risk",
              "extracted_mass", "mass_violation", "marginal_excess",
              "partial_infeasibility", "augmented_infeasibility", "converged"]
    widths = {f: max(len(f), *(len(f"{getattr(r, f):.6g}") if isinstance(getattr(r, f), float)
                              else len(str(getattr(r, f))) for r in rows)) for f in fields}
    print("  ".join(f"{f:<{widths[f]}}" for f in fields))
    print("  ".join("-" * widths[f] for f in fields))
    for r in rows:
        values = []
        for f in fields:
            v = getattr(r, f)
            text = f"{v:.6g}" if isinstance(v, float) else str(v)
            values.append(f"{text:<{widths[f]}}")
        print("  ".join(values))


def load_results_from_csv(csv_path: str) -> list[Result]:
    """Load results from a previously generated CSV file."""
    rows: list[Result] = []
    with open(csv_path, mode="r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                Result(
                    solver=row["solver"],
                    n=int(row["n"]),
                    marginals=int(row["marginals"]),
                    mass=float(row["mass"]),
                    eta=float(row["eta"]),
                    tolerance=float(row["tolerance"]),
                    iterations=int(row["iterations"]),
                    seconds=float(row["seconds"]),
                    converged=row["converged"].strip().lower() in ("true", "1", "yes"),
                    robust_risk=float(row["robust_risk"]),
                    transport_objective=float(row["transport_objective"]),
                    extracted_mass=float(row["extracted_mass"]),
                    mass_violation=float(row["mass_violation"]),
                    marginal_excess=float(row["marginal_excess"]),
                    partial_infeasibility=float(row["partial_infeasibility"]),
                    max_constraint_residual=float(row["max_constraint_residual"]),
                    augmented_infeasibility=float(row.get("augmented_infeasibility", 0.0)),
                )
            )
    return rows


def plot_comparison(
    rows: Sequence[Result],
    output_path: str = "robust_risk_comparison.png",
    show: bool = True,
) -> None:
    """Generate and display/save publication-quality parameter comparison plots.

    Compares GreenkhornMMPOT and GreenkhornMMOT across parameter sweeps
    (regularization eta, transported mass s) on Robust Risk, Infeasibility,
    Iterations, and Runtime.
    """
    if not rows:
        print("No results to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

    solvers = sorted(list(set(r.solver for r in rows)))
    masses = sorted(list(set(r.mass for r in rows)))
    etas = sorted(list(set(r.eta for r in rows)))

    # Determine primary sweep parameter for x-axis
    if len(etas) > 1:
        x_param = "eta"
        x_label = r"Entropic Regularization $\eta$"
        group_param = "mass"
        group_values = masses
        group_label_fn = lambda m: f"$s={m:.2g}$"
    elif len(masses) > 1:
        x_param = "mass"
        x_label = r"Transported Mass $s$"
        group_param = "eta"
        group_values = etas
        group_label_fn = lambda e: f"$\\eta={e:.2g}$"
    else:
        x_param = "eta"
        x_label = r"Entropic Regularization $\eta$"
        group_param = "mass"
        group_values = masses
        group_label_fn = lambda m: f"$s={m:.2g}$"

    # Pre-defined high-contrast styles per solver
    solver_styles = {
        "GreenkhornMMPOT": {
            "linestyle": "-",
            "marker": "o",
            "color": "#1f77b4",  # Rich Blue
            "label_prefix": "MMPOT (Direct)",
        },
        "GreenkhornMMOT": {
            "linestyle": "--",
            "marker": "s",
            "color": "#d62728",  # Crimson Red
            "label_prefix": "MMOT (Augmented)",
        },
    }

    # If multiple masses are present, generate variations
    multi_group_palette = plt.cm.get_cmap("tab10") if hasattr(plt.cm, "get_cmap") else plt.colormaps["tab10"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=100)
    n_dim = rows[0].n
    n_marg = rows[0].marginals
    title_suffix = f", $s={masses[0]:.2g}$" if len(masses) == 1 else ""
    fig.suptitle(
        f"Robust Risk & Solver Parameter Comparison ($n={n_dim}$, $m={n_marg}$ marginals{title_suffix})",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    metrics = [
        ("robust_risk", r"Robust Risk Objective $\langle \mathrm{payoff}, X \rangle / s$", False, axes[0, 0]),
        ("partial_infeasibility", r"Partial Infeasibility $\|X\mathbf{1} - s\| + \sum (r_k - p_k)_+$", True, axes[0, 1]),
        ("seconds", "Computation Time (seconds)", True, axes[1, 0]),
        ("iterations", "Iterations to Convergence", False, axes[1, 1]),
    ]

    for metric_attr, metric_title, is_log_y, ax in metrics:
        for g_idx, g_val in enumerate(group_values):
            for s_idx, solver in enumerate(solvers):
                default_color = "#2ca02c" if s_idx > 1 else ("#1f77b4" if s_idx == 0 else "#d62728")
                style = solver_styles.get(
                    solver,
                    {"linestyle": "-.", "marker": "^", "color": default_color, "label_prefix": solver}
                )

                matching_rows = [
                    r for r in rows
                    if r.solver == solver and getattr(r, group_param) == g_val
                ]
                matching_rows.sort(key=lambda r: getattr(r, x_param))
                if not matching_rows:
                    continue

                xs = [getattr(r, x_param) for r in matching_rows]
                raw_ys = [getattr(r, metric_attr) for r in matching_rows]
                # Safe lower bound for log scale to avoid log(0)
                ys = [max(float(y), 1e-16) if is_log_y else float(y) for y in raw_ys]

                # Assign distinct color per solver when 1 group value, or vary by group
                if len(group_values) == 1:
                    line_color = style["color"]
                    label = style["label_prefix"]
                else:
                    line_color = multi_group_palette((g_idx * len(solvers) + s_idx) % 10)
                    label = f"{style['label_prefix']} ({group_label_fn(g_val)})"

                ax.plot(
                    xs,
                    ys,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    color=line_color,
                    linewidth=2.0,
                    markersize=6,
                    label=label,
                )

                # Mark any non-converged points with an 'x'
                non_converged_xs = [getattr(r, x_param) for r in matching_rows if not r.converged]
                non_converged_ys = [
                    max(float(getattr(r, metric_attr)), 1e-16) if is_log_y else float(getattr(r, metric_attr))
                    for r in matching_rows if not r.converged
                ]
                if non_converged_xs:
                    ax.scatter(
                        non_converged_xs,
                        non_converged_ys,
                        marker="x",
                        color="red",
                        s=50,
                        zorder=5,
                        label="Not Converged" if (g_idx == 0 and s_idx == 0) else None,
                    )

        ax.set_title(metric_title, fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(metric_title.split("(")[0].strip(), fontsize=9)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)

        if is_log_y:
            ax.set_yscale("log")
        if x_param == "eta" and len(etas) > 1 and max(etas) / (min(etas) + 1e-12) >= 4:
            ax.set_xscale("log")

        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        print(f"\nSaved parameter comparison plot to: {out_file.resolve()}")

    if show:
        print("Displaying matplotlib plot window...")
        plt.show()
    plt.close(fig)


def parse_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimension", "-n", type=int, default=200)
    ap.add_argument("--marginals", "-m", type=int, default=3)
    ap.add_argument("--masses", type=parse_floats, default=parse_floats("0.5"),
                    help="comma-separated transported masses (ES tail probabilities)")
    ap.add_argument("--etas", type=parse_floats, default=parse_floats("0.005,0.01"))
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--max-iterations", type=int, default=20000)
    ap.add_argument("--dummy-penalty", type=float, default=1.0,
                    help="finite cost on forbidden dummy patterns")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", type=str, default="", help="optional output CSV path")
    ap.add_argument("--plot", type=str, default="robust_risk_comparison.png",
                    help="path to save parameter comparison plot (default: robust_risk_comparison.png)")
    ap.add_argument("--no-show", action="store_true", help="do not open interactive GUI window (save image only)")
    ap.add_argument("--no-plot", action="store_true", help="disable plotting completely")
    ap.add_argument("--plot-csv", type=str, default="",
                    help="load results from an existing CSV file and plot without re-running solvers")
    args = ap.parse_args()

    if args.plot_csv:
        rows = load_results_from_csv(args.plot_csv)
        print_table(rows)
        if not args.no_plot:
            plot_comparison(rows, output_path=args.plot, show=not args.no_show)
        return

    if args.dimension < 2 or args.marginals < 2:
        ap.error("dimension and marginals must both be at least 2")
    if any(not 0.0 < s < 1.0 for s in args.masses):
        ap.error("every transported mass must lie strictly between 0 and 1")
    if any(e <= 0.0 for e in args.etas):
        ap.error("eta must be positive")

    p, payoff, cost = synthetic_problem(args.dimension, args.marginals, args.seed)
    rows: list[Result] = []
    for s, eta in itertools.product(args.masses, args.etas):
        x, it, ok, elapsed = greenkhorn_mmpot(
            cost, p, s, eta, args.tolerance, args.max_iterations)
        rows.append(make_result("GreenkhornMMPOT", x, p, payoff, cost, s, eta,
                                args.tolerance, it, elapsed, ok))

        ext_c, ext_p = dummy_extension(cost, p, s, args.dummy_penalty)
        y, it, ok, elapsed, aug_inf = greenkhorn_mmot(
            ext_c, ext_p, eta, args.tolerance, args.max_iterations)
        real = (slice(0, args.dimension),) * args.marginals
        extracted = y[real].copy()
        rows.append(make_result("GreenkhornMMOT", extracted, p, payoff, cost, s, eta,
                                args.tolerance, it, elapsed, ok, aug_inf))

    print_table(rows)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)
        print(f"\nWrote {args.csv}")

    if not args.no_plot:
        plot_comparison(rows, output_path=args.plot, show=not args.no_show)


if __name__ == "__main__":
    main()

