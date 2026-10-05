"""Figure 3: the penalty levels the hierarchy of centers.

Four-hill torus, gamma = 1.5. Continuation downward in kappa (0.1, 0.05, 0.02), each
run warm-started from the previous equilibrium at omega = 0.5. The figure shows
kappa = 0.1 and kappa = 0.02; kappa = 0.05 is the intermediate step of the chain
and is archived with it. Hill-associated density maxima are recorded per hill.
"""
import json

import numpy as np

from common import ARCHIVE, TORUS_N, hills_capacity, torus_kernel, torus_centers, margin, report, save, solve

GAMMA = 1.5
KAPPAS = (0.1, 0.05, 0.02)
HILLS = ((14, 14), (14, 46), (46, 20), (44, 48))   # bump centers, tallest first


def hill_maxima(density: np.ndarray) -> list[float]:
    d = density.reshape(TORUS_N, TORUS_N)
    out = []
    for cy, cx in HILLS:
        ys = [(cy + k) % TORUS_N for k in range(-5, 6)]
        xs = [(cx + k) % TORUS_N for k in range(-5, 6)]
        out.append(float(d[np.ix_(ys, xs)].max()))
    return out


def main() -> None:
    R = hills_capacity()
    np.save(f"{ARCHIVE}/fig3_hills_R.npy", R)
    K = torus_kernel()
    summary = {"hill_capacities": [float(R.reshape(TORUS_N, TORUS_N)[cy, cx]) for cy, cx in HILLS], "runs": {}}
    print("Figure 3 (four-hill torus, gamma=1.5):", flush=True)
    prev = None
    for kappa in KAPPAS:
        r = solve(K, R, GAMMA, kappa, start=prev, omega=0.5)
        prev = np.array(r.mass, copy=True)
        name = f"fig3_hills_g1.5_k{kappa:g}"
        d = r.mass / R
        summary["runs"][name] = {"kappa": kappa, "status": r.status.name, "iterations": r.iterations,
                                 "final_step_norm": r.final_step_norm, "max_density": float(d.max()),
                                 "min_density": float(d.min()), "centers": torus_centers(d),
                                 "hill_maxima": hill_maxima(d), **margin(r)}
        report(name, r, R)
        save(name, r)
    with open(f"{ARCHIVE}/fig3_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
