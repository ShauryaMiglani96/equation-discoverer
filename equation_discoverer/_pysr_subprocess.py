"""
_pysr_subprocess.py

PySR worker called by app.py via subprocess.run().

Running PySR in a fresh process avoids Julia's PerThreadCache BoundsErrors
that occur when juliacall is invoked from Streamlit's background Python threads
(Julia's Threads.threadid() can return IDs outside the cache's size).

Protocol
--------
stdin  : JSON object with keys:
           X, y           — 2-D and 1-D lists (floats)
           feature_names  — list[str]
           result_path    — absolute path to write the pickled PySRRegressor
           output_dir     — PySR output_directory
           niterations, maxsize, include_trig, parsimony
stdout : "ok" on success
stderr : error message / traceback on failure (caller checks returncode)
"""
import os
os.environ["PYTHON_JULIACALL_THREADS"] = "1"   # single-threaded; serial PySR needs no more

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pysr import PySRRegressor


def main() -> None:
    params = json.loads(sys.stdin.read())

    X             = np.array(params["X"],  dtype=float)
    y             = np.array(params["y"],  dtype=float)
    feature_names = params["feature_names"]
    result_path   = params["result_path"]
    output_dir    = params["output_dir"]
    niterations   = int(params["niterations"])
    maxsize       = int(params["maxsize"])
    include_trig  = bool(params["include_trig"])
    parsimony     = float(params.get("parsimony", 0.01))

    unary = ["sqrt", "exp", "log"] + (["sin", "cos"] if include_trig else [])

    nested = {
        "exp":  {"exp": 0},
        "sqrt": {"sqrt": 0},
    }
    if include_trig:
        nested["sin"] = {"sin": 0, "cos": 0}
        nested["cos"] = {"sin": 0, "cos": 0}

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=unary,
        constraints={"^": (-1, 1)},
        nested_constraints=nested,
        niterations=niterations,
        populations=15,
        maxsize=maxsize,
        parsimony=parsimony,
        model_selection="score",
        parallelism="serial",
        output_directory=output_dir,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y, variable_names=feature_names)

    with open(result_path, "wb") as f:
        pickle.dump(model, f)

    print("ok")


if __name__ == "__main__":
    main()
