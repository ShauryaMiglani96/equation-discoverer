"""
app.py  —  Equation Discoverer

Streamlit web app for discovering symbolic equations from uploaded CSV data.

Two modes:
  Algebraic  (PySR)   — finds  y = f(x₁, x₂, …)  from any tabular dataset
  Differential (SINDy) — finds  ẋ = f(x)  from time-series data

Usage
-----
    cd equation_discoverer
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from sindy_discover import PYSINDY_AVAILABLE, discover_ode, simulate_trajectory

_OUTPUT_DIR = Path(__file__).parent / "outputs" / "app_run"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_PYSR_SCRIPT = Path(__file__).parent / "_pysr_subprocess.py"


# ── PySR helpers ───────────────────────────────────────────────────────────

def _run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    niterations: int,
    maxsize: int,
    include_trig: bool,
    parsimony: float = 0.01,
):
    """
    Runs PySR in a subprocess isolated from Streamlit's Python threads.

    Julia's PerThreadCache is sized to Threads.nthreads() but
    Threads.threadid() returns IDs from Streamlit's background threads,
    causing BoundsErrors.  A fresh subprocess has no extra threads.
    """
    fd, result_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        params = {
            "X":             X.tolist(),
            "y":             y.tolist(),
            "feature_names": feature_names,
            "result_path":   result_path,
            "output_dir":    str(_OUTPUT_DIR),
            "niterations":   niterations,
            "maxsize":       maxsize,
            "include_trig":  include_trig,
            "parsimony":     parsimony,
        }
        proc = subprocess.run(
            [sys.executable, str(_PYSR_SCRIPT)],
            input=json.dumps(params),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or f"PySR subprocess exited with code {proc.returncode}"
            raise RuntimeError(err)

        with open(result_path, "rb") as f:
            model = pickle.load(f)
        return model
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def _show_pysr_results(
    model,
    y_train: np.ndarray, y_pred_train: np.ndarray,
    y_test:  np.ndarray, y_pred_test:  np.ndarray,
    target_name: str,
) -> None:
    try:
        best_eq = str(model.sympy())
    except Exception:
        best_eq = str(
            model.equations_.loc[int(model.equations_["score"].idxmax()), "equation"]
        )

    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("Best Equation")
        st.code(best_eq)
    with col2:
        st.metric("Train R²", f"{r2_train:.4f}")
    with col3:
        st.metric("Test R²",  f"{r2_test:.4f}")

    gap = r2_train - r2_test
    if gap > 0.05:
        st.warning(
            f"Possible overfitting: train R²={r2_train:.4f} vs test R²={r2_test:.4f} "
            f"(gap {gap:.4f}).  Try raising Parsimony or lowering Max expression size."
        )

    st.subheader("Pareto Front")
    eqs = model.equations_[["complexity", "loss", "score", "equation"]].copy()
    eqs.columns = ["Complexity", "Loss", "Score", "Equation"]
    eqs["Loss"]  = eqs["Loss"].map("{:.4e}".format)
    eqs["Score"] = eqs["Score"].map("{:.4f}".format)
    st.dataframe(eqs, use_container_width=True)

    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    lo = min(all_true.min(), all_pred.min())
    hi = max(all_true.max(), all_pred.max())
    m  = (hi - lo) * 0.05
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", lw=1.2, label="y = x")
    ax.scatter(y_train, y_pred_train, s=14, alpha=0.5, color="steelblue",
               linewidths=0, label=f"Train  R²={r2_train:.4f}")
    ax.scatter(y_test,  y_pred_test,  s=18, alpha=0.8, color="tomato",
               linewidths=0, label=f"Test   R²={r2_test:.4f}")
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title("Actual vs Predicted (train = blue, test = red)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── SINDy helpers ──────────────────────────────────────────────────────────

def _show_sindy_results(
    equations: list[str],
    r2: float,
    t: np.ndarray,
    X_actual: np.ndarray,
    X_sim: np.ndarray | None,
    state_names: list[str],
) -> None:
    st.subheader("Discovered Differential Equations")
    for name, eq in zip(state_names, equations):
        st.code(f"d({name})/dt = {eq}")
    st.metric("R² (derivative fit)", f"{r2:.6f}")

    if X_sim is None:
        st.warning(
            "Trajectory simulation diverged — the discovered equations may need "
            "threshold tuning. Try lowering the sparsification threshold."
        )

    n = len(state_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for i, (name, eq) in enumerate(zip(state_names, equations)):
        ax = axes[i, 0]
        ax.plot(t, X_actual[:, i], lw=1.5, color="steelblue", label="Actual")
        if X_sim is not None:
            n_pts = min(len(t), len(X_sim))
            ax.plot(
                t[:n_pts], X_sim[:n_pts, i],
                lw=1.5, color="tomato", linestyle="--", label="Simulated",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(name)
        ax.set_title(f"d({name})/dt = {eq}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Equation Discoverer", layout="wide")
st.title("Equation Discoverer")
st.markdown(
    "Upload a CSV and discover its governing equation.  "
    "**Algebraic mode** (PySR) finds `y = f(x₁, x₂, …)` from any tabular data.  "
    "**Differential mode** (SINDy) finds `ẋ = f(x)` from time-series data."
)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("1  Upload")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV file to get started.")
        st.stop()

    df = pd.read_csv(uploaded)

    # Clear stale results when the file changes
    if st.session_state.get("_file") != uploaded.name:
        st.session_state.pop("result", None)
        st.session_state["_file"] = uploaded.name

    n_missing = int(df.isnull().sum().sum())
    if n_missing:
        st.warning(f"Dropped {n_missing} missing value(s).")
        df = df.dropna().reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        st.stop()

    st.divider()
    st.header("2  Mode")
    use_sindy = st.toggle(
        "Differential equations (SINDy)",
        value=False,
        help="OFF: algebraic  y = f(x)   |   ON: differential  ẋ = f(x) from time-series",
    )

    # Clear stale results when mode changes
    if st.session_state.get("_mode") != use_sindy:
        st.session_state.pop("result", None)
        st.session_state["_mode"] = use_sindy

    if use_sindy and not PYSINDY_AVAILABLE:
        st.error("Run  `pip install pysindy`  to enable SINDy mode.")
        st.stop()

    st.divider()
    st.header("3  Columns")

    if not use_sindy:
        # ── Algebraic (PySR) ───────────────────────────────────────────────
        feature_cols = st.multiselect(
            "Feature columns  (x)",
            options=num_cols,
            default=num_cols[:-1],
        )
        remaining = [c for c in num_cols if c not in feature_cols]
        target_col = st.selectbox(
            "Target column  (y)",
            options=remaining if remaining else num_cols,
        )
        ready = bool(feature_cols) and target_col not in feature_cols

        st.divider()
        with st.expander("PySR parameters"):
            niterations  = st.slider("Iterations", 10, 200, 40, 10)
            maxsize      = st.slider("Max expression size", 10, 40, 20, 5)
            parsimony    = st.slider(
                "Parsimony (complexity penalty)", 0.001, 0.1, 0.01, 0.001,
                format="%.3f",
                help="Higher → prefers simpler equations. Default 0.01 (3× PySR default). "
                     "Raise to 0.05–0.1 if you see overfitting.",
            )
            include_trig = st.checkbox("Include sin / cos", value=True)

    else:
        # ── Differential (SINDy) ──────────────────────────────────────────
        time_col = st.selectbox("Time column", options=num_cols)
        state_cols = st.multiselect(
            "State variable columns",
            options=[c for c in num_cols if c != time_col],
            default=[c for c in num_cols if c != time_col][:2],
        )
        st.caption("Rows must be in ascending time order.")
        ready = bool(state_cols)

        st.divider()
        with st.expander("SINDy parameters"):
            poly_degree    = st.slider("Polynomial degree", 2, 5, 3)
            include_trig_s = st.checkbox("Include sin / cos", value=False)
            threshold      = st.slider(
                "Sparsification threshold", 0.001, 0.5, 0.05, 0.005,
                help="Higher → fewer terms (sparser model).",
            )

    st.divider()
    run_btn = st.button(
        "Run", type="primary", disabled=not ready, use_container_width=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────

tab_data, tab_results = st.tabs(["Data Preview", "Results"])

with tab_data:
    st.write(f"**{df.shape[0]}** rows  ×  **{df.shape[1]}** columns")
    st.dataframe(df.head(50), use_container_width=True)
    with st.expander("Summary statistics"):
        st.dataframe(df.describe().T.round(4), use_container_width=True)

with tab_results:
    # ── Run computation when button clicked ────────────────────────────────
    if run_btn:
        if not use_sindy:
            X = df[feature_cols].values.astype(float)
            y = df[target_col].values.astype(float)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            with st.spinner("Running PySR — may take a few minutes on first run..."):
                try:
                    model = _run_pysr(
                        X_train, y_train, feature_cols,
                        niterations, maxsize, include_trig, parsimony,
                    )
                    st.session_state["result"] = {
                        "mode":         "pysr",
                        "model":        model,
                        "y_train":      y_train,
                        "y_pred_train": model.predict(X_train),
                        "y_test":       y_test,
                        "y_pred_test":  model.predict(X_test),
                        "target":       target_col,
                        "features":     feature_cols,
                    }
                except Exception as exc:
                    st.error(f"PySR failed: {exc}")

        else:
            t      = df[time_col].values.astype(float)
            X_st   = df[state_cols].values.astype(float)
            with st.spinner("Running SINDy..."):
                try:
                    sindy_model, equations, r2 = discover_ode(
                        t, X_st, state_cols,
                        poly_degree=poly_degree,
                        include_trig=include_trig_s,
                        threshold=threshold,
                    )
                    X_sim = simulate_trajectory(sindy_model, X_st[0], t)
                    st.session_state["result"] = {
                        "mode":        "sindy",
                        "equations":   equations,
                        "r2":          r2,
                        "t":           t,
                        "X_actual":    X_st,
                        "X_sim":       X_sim,
                        "state_names": state_cols,
                    }
                except Exception as exc:
                    st.error(f"SINDy failed: {exc}")

    # ── Display results (cached or freshly computed) ───────────────────────
    res = st.session_state.get("result")
    if res is None:
        st.info("Configure inputs in the sidebar, then click **Run**.")
    elif res["mode"] == "pysr":
        _show_pysr_results(
            res["model"],
            res["y_train"], res["y_pred_train"],
            res["y_test"],  res["y_pred_test"],
            res["target"],
        )
    elif res["mode"] == "sindy":
        _show_sindy_results(
            res["equations"], res["r2"],
            res["t"], res["X_actual"], res["X_sim"],
            res["state_names"],
        )