"""
discover_equations.py

Runs PySR symbolic regression on every physical-law dataset defined in
physical_datasets.py.  For each dataset it:
  1. Fits a PySRRegressor on the raw (SI-unit) data, with per-dataset
     rescaling so that all numbers PySR sees are O(1–1000).
  2. Prints a formatted Pareto-front table (Complexity / MSE Loss / Score /
     Discovered Equation).
  3. Prints the best equation in SymPy form alongside the R² score.
  4. Builds a matplotlib figure (2 × 3 grid) of actual vs predicted values.

Usage
-----
    cd equation_discoverer
    python discover_equations.py
"""


from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from pysr import PySRRegressor
from physical_datasets import DATASETS, get_dataset

# ── Output folder ──────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_OUTPUT_DIR = _HERE / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)

# ── Shared PySR hyperparameters ────────────────────────────────────────────
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

# ── Per-dataset rescaling constants ───────────────────────────────────────
_AU   = 1.496e11        # metres per astronomical unit
_MSUN = 1.989e30        # kg per solar mass
_YR   = 365.25 * 24 * 3600  # seconds per year


def _preprocess(name: str, df, meta: dict):
    """
    Return (df, meta, weights) with dataset-specific unit rescaling.

    projectile_range  — theta stored in degrees; formula needs radians.
                        Converting removes the hidden π/180 constant from
                        inside sin(), letting PySR find sin(2·θ) directly.

    coulomb_law       — charges in Coulombs are O(1e-9 – 1e-6); Coulomb's
                        constant k ≈ 9e9 compounds the scale problem and F
                        spans ~10 orders of magnitude.  Rescaling charges to
                        μC brings k·(1e-6)² ≈ 9e-3 into a reasonable range.
                        weights=1/|F| makes PySR minimise relative error so
                        small-force and large-force samples contribute equally.

    keplers_third_law — raw SI values (metres, kg, seconds) are astronomical
                        in scale.  In natural units (AU, M☉, years) the law
                        collapses to T_yr = sqrt(a_AU³ / M_sol), with all
                        values in [0.1, 256].
    """
    df   = df.copy()
    meta = dict(meta)
    weights = None

    if name == "projectile_range":
        df["theta_rad"] = np.deg2rad(df["theta_deg"])
        df = df.drop(columns=["theta_deg"])
        meta["features"] = ["v0", "theta_rad"]

    elif name == "coulomb_law":
        df["q1_uC"] = df["q1"] * 1e6
        df["q2_uC"] = df["q2"] * 1e6
        df = df.drop(columns=["q1", "q2"])
        meta["features"] = ["q1_uC", "q2_uC", "r"]
        # relative-error weighting: each sample contributes equally regardless
        # of the 10-order-of-magnitude spread in F
        weights = 1.0 / np.abs(df[meta["target"]].values)

    elif name == "keplers_third_law":
        df["a_AU"]  = df["a"]  / _AU
        df["M_sol"] = df["M"]  / _MSUN
        df["T_yr"]  = df["T"]  / _YR
        df = df.drop(columns=["a", "M", "T"])
        meta["features"] = ["a_AU", "M_sol"]
        meta["target"]   = "T_yr"

    return df, meta, weights


# ── Per-dataset fit ────────────────────────────────────────────────────────

def _fit_dataset(
    name: str,
) -> tuple[PySRRegressor, np.ndarray, np.ndarray, dict]:
    """Load raw data, rescale, fit PySR, return (model, y_true, y_pred, meta)."""
    df, meta = get_dataset(name, normalized=False)
    df, meta, weights = _preprocess(name, df, meta)

    X = df[meta["features"]].values
    y = df[meta["target"]].values

    model = PySRRegressor(**_PYSR_KWARGS)
    model.fit(X, y, variable_names=meta["features"], weights=weights)
    y_pred = model.predict(X)
    return model, y, y_pred, meta


# ── Console output ─────────────────────────────────────────────────────────

def _print_results(
    name: str,
    model: PySRRegressor,
    meta: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print Pareto-front table and best equation for one dataset."""
    bar = "═" * 74

    print(f"\n{bar}")
    print(f"  Dataset  : {name.replace('_', ' ').title()}")
    print(f"  True law : {meta['equation']}")
    print(f"  Target   : {meta['target']}     Features : {', '.join(meta['features'])}")
    print(bar)

    eqs = model.equations_.copy()

    # ── Build display table ────────────────────────────────────────────────
    display = eqs[["complexity", "loss", "score", "equation"]].copy()
    display.columns = ["Complexity", "MSE Loss", "Score", "Discovered Equation"]
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

    print("\n  Pareto Front\n")
    print(display.to_string(index=False, max_colwidth=52))

    # ── Best equation ──────────────────────────────────────────────────────
    try:
        best_sympy = str(model.sympy())
    except Exception:
        best_sympy = eqs.loc[best_row, "equation"]

    r2 = r2_score(y_true, y_pred)
    print(f"\n  >>> Best equation (SymPy) : {best_sympy}")
    print(f"  >>> R²                    : {r2:.6f}")
    print(f"{bar}\n")


# ── Plotting ───────────────────────────────────────────────────────────────

def _build_figure(
    results: list[tuple[str, np.ndarray, np.ndarray, PySRRegressor, dict]],
) -> plt.Figure:
    """Return a 2 × 3 figure of actual-vs-predicted scatter plots."""
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11))
    axes_flat = axes.flatten()

    for i, (name, y_true, y_pred, model, meta) in enumerate(results):
        ax = axes_flat[i]
        r2 = r2_score(y_true, y_pred)

        pad = 0.15
        lo = min(y_true.min(), y_pred.min()) - pad
        hi = max(y_true.max(), y_pred.max()) + pad
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.2,
                linestyle="--", label="Perfect fit  (y = x)")

        ax.scatter(y_true, y_pred, s=12, alpha=0.45,
                   color="steelblue", linewidths=0, label="Predictions")

        try:
            eq_str = str(model.sympy())
        except Exception:
            try:
                best_complexity = int(model.get_best()["complexity"])
                eqs = model.equations_
                match = eqs.index[eqs["complexity"] == best_complexity]
                best_row = int(match[0]) if len(match) else int(eqs["score"].idxmax())
            except Exception:
                best_row = int(model.equations_["score"].idxmax())
            eq_str = model.equations_.loc[best_row, "equation"]

        short_eq = eq_str if len(eq_str) <= 48 else eq_str[:45] + "..."

        ax.set_title(
            f"{name.replace('_', ' ').title()}\n"
            f"True law : {meta['equation']}\n"
            f"Found    : {short_eq}\n"
            f"R² = {r2:.4f}",
            fontsize=8.5,
            loc="left",
            pad=6,
        )
        ax.set_xlabel(f"Actual  {meta['target']}", fontsize=8)
        ax.set_ylabel(f"Predicted  {meta['target']}", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Symbolic Regression — Actual vs Predicted",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    names = list(DATASETS.keys())
    results: list[tuple[str, np.ndarray, np.ndarray, PySRRegressor, dict]] = []

    for idx, name in enumerate(names, 1):
        print(f"\n{'─' * 74}")
        print(f"  [{idx}/{len(names)}]  Running PySR on: {name}")
        print(f"{'─' * 74}")
        model, y_true, y_pred, meta = _fit_dataset(name)
        _print_results(name, model, meta, y_true, y_pred)
        results.append((name, y_true, y_pred, model, meta))

    fig = _build_figure(results)
    plot_path = _OUTPUT_DIR / "actual_vs_predicted.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {plot_path}\n")
    plt.show()


if __name__ == "__main__":
    main()
