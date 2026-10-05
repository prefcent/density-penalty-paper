"""Figure 1: what the penalty is for.

Ring, gamma = 2, the model's own map (omega = 1, start a = R). For kappa = 0 and
kappa = 0.0128: a 200-iteration budget, a 1,000-iteration budget, and a run to
tolerance 1e-12. Budget runs report FIXED_BUDGET with their final step; the
tolerance runs show where each budget run was heading.

For every returned state the undamped residual is also evaluated by one further
omega = 1 application of the map. That evaluation is a diagnostic, not an accepted
iteration; the archived state is the run's own.
"""
import json

import numpy as np

from common import ARCHIVE, ring_capacity, ring_kernel, ring_peaks, margin, report, save, solve

GAMMA = 2.0
KAPPAS = (0.0, 0.0128)
BUDGETS = {"b200": dict(max_iter=200, tol=None), "b1000": dict(max_iter=1000, tol=None),
           "tol": dict(max_iter=400_000, tol=1e-12)}


def concentration(mass: np.ndarray) -> dict:
    """Share of all activity in the peak zone, the five largest zones, and within
    10 km of the peak."""
    order = np.argsort(mass)[::-1]
    total = mass.sum()
    peak = int(order[0])
    near = [(peak + k) % mass.size for k in range(-10, 11)]
    return {"peak_zone": peak, "top1": float(mass[peak] / total),
            "top5": float(mass[order[:5]].sum() / total),
            "within_10km": float(mass[near].sum() / total)}


def main() -> None:
    R = ring_capacity()
    K = ring_kernel()
    summary = {}
    print("Figure 1 (ring, gamma=2, omega=1):", flush=True)
    for kappa in KAPPAS:
        for tag, kw in BUDGETS.items():
            name = f"fig1_ring_g2_k{kappa:g}_{tag}"
            r = solve(K, R, GAMMA, kappa, omega=1.0, **kw)
            probe = solve(K, R, GAMMA, kappa, start=np.array(r.mass, copy=True),
                          omega=1.0, tol=None, max_iter=1)
            d = r.mass / R
            row = {"kappa": kappa, "status": r.status.name, "iterations": r.iterations,
                   "final_step_norm": r.final_step_norm,
                   "residual_at_returned_state": probe.final_step_norm,
                   "max_density": float(d.max()), "peaks": len(ring_peaks(d)),
                   **margin(r), **concentration(r.mass)}
            summary[name] = row
            report(name, r, R)
            save(name, r)
    with open(f"{ARCHIVE}/fig1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
