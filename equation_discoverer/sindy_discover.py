"""
sindy_discover.py

SINDy (Sparse Identification of Nonlinear Dynamics) wrapper.
Discovers differential equations of the form  ẋ = f(x)  from time-series data.

The discovered model takes the form:
    [ ẋ₁ ]   [ f₁(x₁, x₂, …) ]
    [ ẋ₂ ] = [ f₂(x₁, x₂, …) ]
    [ …  ]   [ …               ]

where each fᵢ is a sparse linear combination of library terms (polynomials,
optionally sin/cos), identified via STLSQ (Sequentially Thresholded Least Squares).
"""
from __future__ import annotations

import numpy as np

try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False


def discover_ode(
    t: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    poly_degree: int = 3,
    include_trig: bool = False,
    threshold: float = 0.05,
    alpha: float = 0.05,
) -> tuple:
    """
    Fit SINDy to time-series state data.

    Parameters
    ----------
    t             : (n,) time points, sorted ascending
    X             : (n, d) state variable matrix  — one column per variable
    feature_names : name for each column of X
    poly_degree   : degree of polynomial feature library
    include_trig  : if True, add Fourier (sin/cos) terms to the library
    threshold     : STLSQ sparsification threshold  (higher → fewer terms)
    alpha         : STLSQ L₂ regularisation coefficient

    Returns
    -------
    model     : fitted ps.SINDy model
    equations : list[str], one equation string per state variable
    r2        : R² score on the derivative prediction
    """
    if not PYSINDY_AVAILABLE:
        raise ImportError("pysindy is not installed.  Run:  pip install pysindy")

    poly_lib = ps.PolynomialLibrary(degree=poly_degree)

    if include_trig:
        feature_library = poly_lib + ps.FourierLibrary(n_frequencies=1)
    else:
        feature_library = poly_lib

    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
    )
    model.fit(X, t=t, feature_names=feature_names)

    equations = model.equations(precision=4)
    r2 = float(model.score(X, t=t))
    return model, equations, r2


def simulate_trajectory(
    model,
    x0: np.ndarray,
    t: np.ndarray,
) -> np.ndarray | None:
    """
    Integrate the discovered ODE forward from initial state x0.
    Returns None if pysindy is unavailable or the integration diverges.
    """
    if not PYSINDY_AVAILABLE:
        return None
    try:
        return model.simulate(x0, t)
    except Exception:
        return None
