"""Fase offline: DOE Latin Hypercube sui parametri di progetto + lancio delle simulazioni FEM.

Uso:  python run_doe.py [n_campioni] [seed]
Riprendibile: i casi gia' presenti in data/ vengono saltati. I casi falliti finiscono in data/failed.txt.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import fem

DATA = Path(__file__).parent / "data"


def make_design(n, seed):
    lo = np.array([p[1] for p in fem.PARAMS])
    hi = np.array([p[2] for p in fem.PARAMS])
    return qmc.scale(qmc.LatinHypercube(d=len(fem.PARAMS), seed=seed).random(n), lo, hi)


def main(n=150, seed=0):
    DATA.mkdir(exist_ok=True)
    design = make_design(n, seed)
    np.save(DATA / "design.npy", design)

    mapdl = fem.start_mapdl(DATA / "_mapdl")
    t0 = time.time()
    try:
        for i, mu in enumerate(design):
            out = DATA / f"case_{i:04d}.npz"
            if out.exists():
                continue
            try:
                np.savez(out, **fem.run_case(mapdl, mu))
            except Exception:
                with open(DATA / "failed.txt", "a") as f:
                    f.write(f"{i} {mu.tolist()}\n{traceback.format_exc()}\n")
                mapdl.exit(force=True)
                mapdl = fem.start_mapdl(DATA / "_mapdl")
            print(f"[{i + 1}/{n}] mu={np.round(mu, 2)}  t={time.time() - t0:.0f}s", flush=True)
    finally:
        mapdl.exit(force=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
