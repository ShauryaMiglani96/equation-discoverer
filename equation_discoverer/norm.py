"""
norm.py

Runs PySR on z-scored data, then algebraically converts the discovered
equation back to physical (SI-unit) form using the stored normalization stats.

If PySR finds f(x0_norm, x1_norm, ...) in normalised space, the physical
equation is recovered as:
    y = f((x0 - μ_x0)/σ_x0, (x1 - μ_x1)/σ_x1, ...) * σ_y + μ_y

Usage
-----
    cd equation_discoverer
    python norm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))

from pysr import PySRRegressor
from physical_datasets import DATASETS, get_dataset

_HERE = Path(__file__).parent
_OUTPUT_DIR = _HERE / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)

_PYSR_KWARGS = dict(
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "sin", "cos", "exp", "log"],
    constraints={"^": (-1, 1)},
    nested_constraints={
        "sin":  {"sin": 0, "cos": 0},
        "cos":  {"sin": 0, "cos": 0},
        "exp":  {"exp": 0},
        "sqrt": {"sqrt": 0},
    },
    niterations=40,
    populations=15,
    maxsize=20,
    parsimony=0.01,
    model_selection="score",
    parallelism="serial",
)


# ── Denormalization ────────────────────────────────────────────────────────

def _denormalize_equation(
    model: PySRRegressor,
    meta: dict,
    norm_stats: dict,
) -> sp.Expr | None:
    """
    Convert the best SymPy expression (in z-scored variable space) back to
    physical units.

    For each feature x with stats (μ, σ):
        substitute  x_norm  →  (x - μ) / σ
    Then undo the target z-scoring:
        y_physical = expr_after_subs * σ_y + μ_y
    """
    try:
        norm_expr = model.sympy()
    except Exception:
        return None

    # Build substitution: normalized symbol → (raw_symbol - mean) / std
    # SymPy's subs() is a single-pass replacement, so using the same symbol
    # on both sides is safe — the substituted expression is not re-expanded.
    subs = {
        sp.Symbol(feat): (sp.Symbol(feat) - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
        for feat in meta["features"]
    }

    t_stats = norm_stats[meta["target"]]
    denorm_expr = norm_expr.subs(subs) * t_stats["std"] + t_stats["mean"]

    try:
        denorm_expr = sp.simplify(denorm_expr)
    except Exception:
        pass

    return denorm_expr


# ── Per-dataset fit ────────────────────────────────────────────────────────

def _fit_dataset(
    name: str,
) -> tuple[PySRRegressor, np.ndarray, np.ndarray, dict, dict]:
    """Fit PySR on normalised data; return model, y_true, y_pred, meta, norm_stats."""
    df, meta = get_dataset(name, normalized=True)
    norm_stats = meta["normalization_stats"]

    X = df[meta["features"]].values
    y = df[meta["target"]].values

    model = PySRRegressor(**_PYSR_KWARGS)
    model.fit(X, y, variable_names=meta["features"])
    y_pred = model.predict(X)
    return model, y, y_pred, meta, norm_stats


# ── Console output ─────────────────────────────────────────────────────────

def _print_results(
    name: str,
    model: PySRRegressor,
    meta: dict,
    norm_stats: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    bar = "═" * 74

    print(f"\n{bar}")
    print(f"  Dataset  : {name.replace('_', ' ').title()}")
    print(f"  True law : {meta['equation']}")
    print(f"  Target   : {meta['target']}     Features : {', '.join(meta['features'])}")
    print(bar)

    eqs = model.equations_.copy()

    display = eqs[["complexity", "loss", "score", "equation"]].copy()
    display.columns = ["Complexity", "MSE Loss", "Score", "Equation (normalised)"]
    display["MSE Loss"] = display["MSE Loss"].map("{:.4e}".format)
    display["Score"]    = display["Score"].map("{:.4f}".format)
    display.index = range(len(display))

    try:
        best_complexity = int(model.get_best()["complexity"])
        match = eqs.index[eqs["complexity"] == best_complexity]
        best_row = int(match[0]) if len(match) else int(eqs["score"].idxmax())
    except Exception:
        best_row = int(eqs["score"].idxmax())

    markers = ["   " for _ in range(len(display))]
    markers[best_row] = ">>>"
    display.insert(0, "  ", markers)

    print("\n  Pareto Front (z-scored space)\n")
    print(display.to_string(index=False, max_colwidth=52))

    # Best equation in normalised space
    try:
        norm_str = str(model.sympy())
    except Exception:
        norm_str = eqs.loc[best_row, "equation"]

    # Best equation converted back to physical units
    denorm_expr = _denormalize_equation(model, meta, norm_stats)
    denorm_str  = str(denorm_expr) if denorm_expr is not None else "conversion failed"

    r2 = r2_score(y_true, y_pred)
    print(f"\n  >>> Equation (normalised) : {norm_str}")
    print(f"  >>> Equation (physical)   : {denorm_str}")
    print(f"  >>> R² (normalised data)  : {r2:.6f}")
    print(f"{bar}\n")


# ── Plotting ───────────────────────────────────────────────────────────────

def _build_figure(
    results: list[tuple[str, np.ndarray, np.ndarray, PySRRegressor, dict, dict]],
) -> plt.Figure:
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11))
    axes_flat = axes.flatten()

    for i, (name, y_true, y_pred, model, meta, norm_stats) in enumerate(results):
        ax = axes_flat[i]
        r2 = r2_score(y_true, y_pred)

        pad = 0.05
        lo = min(y_true.min(), y_pred.min()) - pad
        hi = max(y_true.max(), y_pred.max()) + pad
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.2,
                linestyle="--", label="Perfect fit  (y = x)")
        ax.scatter(y_true, y_pred, s=12, alpha=0.45,
                   color="steelblue", linewidths=0, label="Predictions")

        # Show the physical-unit equation in the subplot title
        denorm_expr = _denormalize_equation(model, meta, norm_stats)
        if denorm_expr is not None:
            eq_str = str(denorm_expr)
        else:
            try:
                eq_str = str(model.sympy()) + "  [normalised]"
            except Exception:
                eq_str = model.equations_.loc[
                    int(model.equations_["score"].idxmax()), "equation"
                ] + "  [normalised]"

        short_eq = eq_str if len(eq_str) <= 52 else eq_str[:49] + "..."

        ax.set_title(
            f"{name.replace('_', ' ').title()}\n"
            f"True law : {meta['equation']}\n"
            f"Found    : {short_eq}\n"
            f"R² = {r2:.4f}",
            fontsize=8.5,
            loc="left",
            pad=6,
        )
        ax.set_xlabel(f"Actual {meta['target']} (normalised)", fontsize=8)
        ax.set_ylabel(f"Predicted {meta['target']} (normalised)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Symbolic Regression — Actual vs Predicted  (z-scored data, equations shown in physical units)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    names = list(DATASETS.keys())
    results: list[tuple[str, np.ndarray, np.ndarray, PySRRegressor, dict, dict]] = []

    for idx, name in enumerate(names, 1):
        print(f"\n{'─' * 74}")
        print(f"  [{idx}/{len(names)}]  Running PySR on: {name}  (normalised)")
        print(f"{'─' * 74}")
        model, y_true, y_pred, meta, norm_stats = _fit_dataset(name)
        _print_results(name, model, meta, norm_stats, y_true, y_pred)
        results.append((name, y_true, y_pred, model, meta, norm_stats))

    fig = _build_figure(results)
    plot_path = _OUTPUT_DIR / "actual_vs_predicted_norm.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {plot_path}\n")
    plt.show()


if __name__ == "__main__":
    main()
