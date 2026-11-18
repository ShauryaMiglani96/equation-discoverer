"""
Physical equation datasets for symbolic regression discovery.

Each generator samples random inputs across a physically realistic range,
computes the exact output from the known law, and returns both the raw
(SI-unit) DataFrame and a z-score normalised copy (mean=0, std=1).

Available datasets
------------------
simple_pendulum    : T = 2π√(L/g)
ideal_gas_law      : P = nRT/V
projectile_range   : R = v0² sin(2θ) / g
coulomb_law        : F = k·q1·q2 / r²
keplers_third_law  : T = 2π√(a³ / GM)

Typical usage
-------------
    from physical_datasets import get_dataset, DATASETS

    df, meta = get_dataset("simple_pendulum")
    X = df[meta["features"]].values
    y = df[meta["target"]].values

Run this file directly to write all CSVs to ./datasets/:
    python physical_datasets.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Physical constants ─────────────────────────────────────────────────────
_G_GRAVITY: float = 9.81        # m/s²
_R_GAS: float = 8.314           # J/(mol·K)
_K_COULOMB: float = 8.9875e9    # N·m²/C²
_G_NEWTON: float = 6.674e-11    # m³/(kg·s²)
_M_SUN: float = 1.989e30        # kg
_AU: float = 1.496e11           # m  (1 astronomical unit)


# ── Normalisation helper ───────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Z-score normalise every column (mean=0, std=1, ddof=0).

    Returns the normalised DataFrame and a stats dict keyed by column name,
    each with 'mean' and 'std' so callers can invert the transform.
    """
    stats: dict[str, dict[str, float]] = {}
    df_norm = df.copy().astype(float)
    for col in df_norm.columns:
        mean = float(df_norm[col].mean())
        std = float(df_norm[col].std(ddof=0))
        df_norm[col] = (df_norm[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return df_norm, stats


# ── Dataset generators ─────────────────────────────────────────────────────

def simple_pendulum(
    n_samples: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """T = 2π√(L/g)

    Features : L  — pendulum length  (m,  uniform 0.1 – 5.0)
    Target   : T  — period           (s)
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.1, 5.0, n_samples)
    T = 2.0 * np.pi * np.sqrt(L / _G_GRAVITY)
    df = pd.DataFrame({"L": L, "T": T})
    df_norm, stats = _normalize(df)
    return df, df_norm, stats


def ideal_gas_law(
    n_samples: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """P = nRT / V

    Features : n    — amount of gas   (mol, uniform 0.1 – 5.0)
               T_K  — temperature     (K,   uniform 200 – 500)
               V    — volume          (m³,  uniform 0.001 – 0.1)
    Target   : P    — pressure        (Pa)
    """
    rng = np.random.default_rng(seed)
    n = rng.uniform(0.1, 5.0, n_samples)
    T_K = rng.uniform(200.0, 500.0, n_samples)
    V = rng.uniform(0.001, 0.1, n_samples)
    P = n * _R_GAS * T_K / V
    df = pd.DataFrame({"n": n, "T_K": T_K, "V": V, "P": P})
    df_norm, stats = _normalize(df)
    return df, df_norm, stats


def projectile_range(
    n_samples: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """R = v0² sin(2θ) / g

    Features : v0        — launch speed  (m/s, uniform 5 – 50)
               theta_deg — launch angle  (°,   uniform 5 – 85)
    Target   : R         — horizontal range (m)
    """
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(5.0, 50.0, n_samples)
    theta_deg = rng.uniform(5.0, 85.0, n_samples)
    theta_rad = np.deg2rad(theta_deg)
    R = v0 ** 2 * np.sin(2.0 * theta_rad) / _G_GRAVITY
    df = pd.DataFrame({"v0": v0, "theta_deg": theta_deg, "R": R})
    df_norm, stats = _normalize(df)
    return df, df_norm, stats


def coulomb_law(
    n_samples: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """F = k·q1·q2 / r²

    Features : q1 — first charge   (C,  uniform 1e-9 – 1e-6)
               q2 — second charge  (C,  uniform 1e-9 – 1e-6)
               r  — separation     (m,  uniform 0.01 – 1.0)
    Target   : F  — electrostatic force (N)
    """
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(1e-9, 1e-6, n_samples)
    q2 = rng.uniform(1e-9, 1e-6, n_samples)
    r = rng.uniform(0.01, 1.0, n_samples)
    F = _K_COULOMB * q1 * q2 / r ** 2
    df = pd.DataFrame({"q1": q1, "q2": q2, "r": r, "F": F})
    df_norm, stats = _normalize(df)
    return df, df_norm, stats


def keplers_third_law(
    n_samples: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """T = 2π√(a³ / GM)

    Features : a — semi-major axis       (m,  uniform 0.3 – 32 AU)
               M — central body mass     (kg, uniform 0.5 – 2.0 M☉)
    Target   : T — orbital period        (s)
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.3 * _AU, 32.0 * _AU, n_samples)
    M = rng.uniform(0.5 * _M_SUN, 2.0 * _M_SUN, n_samples)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / (_G_NEWTON * M))
    df = pd.DataFrame({"a": a, "M": M, "T": T})
    df_norm, stats = _normalize(df)
    return df, df_norm, stats


# ── Dataset registry ───────────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "simple_pendulum": {
        "generator": simple_pendulum,
        "equation": "T = 2*pi*sqrt(L/g)",
        "target": "T",
        "features": ["L"],
        "description": "Simple pendulum period as a function of string length",
    },
    "ideal_gas_law": {
        "generator": ideal_gas_law,
        "equation": "P = n*R*T / V",
        "target": "P",
        "features": ["n", "T_K", "V"],
        "description": "Ideal gas pressure as a function of n, temperature, and volume",
    },
    "projectile_range": {
        "generator": projectile_range,
        "equation": "R = v0^2 * sin(2*theta) / g",
        "target": "R",
        "features": ["v0", "theta_deg"],
        "description": "Horizontal range of a projectile as a function of speed and angle",
    },
    "coulomb_law": {
        "generator": coulomb_law,
        "equation": "F = k*q1*q2 / r^2",
        "target": "F",
        "features": ["q1", "q2", "r"],
        "description": "Coulomb electrostatic force between two point charges",
    },
    "keplers_third_law": {
        "generator": keplers_third_law,
        "equation": "T = 2*pi*sqrt(a^3 / (G*M))",
        "target": "T",
        "features": ["a", "M"],
        "description": "Kepler's third law: orbital period from semi-major axis and central mass",
    },
}


# ── Public API ─────────────────────────────────────────────────────────────

def get_dataset(
    name: str,
    n_samples: int = 500,
    seed: int = 42,
    normalized: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return a physical-law dataset by name.

    Parameters
    ----------
    name : str
        Key from DATASETS (e.g. ``"simple_pendulum"``).
    n_samples : int
        Number of data points to generate.
    seed : int
        Random seed for reproducibility.
    normalized : bool
        If True (default), return z-score normalised data.
        If False, return raw SI-unit data.

    Returns
    -------
    df : pd.DataFrame
        Feature + target columns.
    meta : dict
        Keys: ``equation``, ``target``, ``features``, ``description``,
        and ``normalization_stats`` (per-column mean & std).
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS)}"
        )
    entry = DATASETS[name]
    df_raw, df_norm, stats = entry["generator"](n_samples=n_samples, seed=seed)
    meta = {
        "equation": entry["equation"],
        "target": entry["target"],
        "features": entry["features"],
        "description": entry["description"],
        "normalization_stats": stats,
    }
    return (df_norm if normalized else df_raw), meta


def save_datasets(
    output_dir: str = "datasets",
    n_samples: int = 500,
    seed: int = 42,
    save_raw: bool = True,
    save_normalized: bool = True,
) -> None:
    """Write all datasets to CSV files.

    Output files
    ------------
    <output_dir>/<name>_raw.csv        — SI units (if save_raw=True)
    <output_dir>/<name>_normalized.csv — z-scored  (if save_normalized=True)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, entry in DATASETS.items():
        df_raw, df_norm, _ = entry["generator"](n_samples=n_samples, seed=seed)
        if save_raw:
            df_raw.to_csv(out / f"{name}_raw.csv", index=False)
            print(f"  wrote {name}_raw.csv")
        if save_normalized:
            df_norm.to_csv(out / f"{name}_normalized.csv", index=False)
            print(f"  wrote {name}_normalized.csv")
    print(f"\nAll datasets saved to '{out.resolve()}'")


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_datasets()
