"""Figure 4: transport cost and the number of centers.

Fixed (gamma, kappa) on each landscape; all line-haul costs scaled by s while the
terminal penalty d0 stays fixed. Continuation in s from slow to fast transport
(10, 3, 1, 0.3), warm-started at omega = 0.5. Torus: gamma = 1.5, kappa = 0.05;
ring: gamma = 0.5, kappa = 0.0128. Center and peak counts use the fixed rule.
"""
import json

import numpy as np

from common import (ARCHIVE, hills_capacity, ring_capacity, ring_kernel, ring_peaks, torus_centers,
                    torus_kernel, margin, report, save, solve)

SCALES = (10.0, 3.0, 1.0, 0.3)
TORUS = dict(gamma=1.5, kappa=0.05)
RING = dict(gamma=0.5, kappa=0.0128)


def main() -> None:
    summary = {"scales": SCALES, "torus": {}, "ring": {}}
    Rt = hills_capacity()
    print("Figure 4, torus (gamma=1.5, kappa=0.05):", flush=True)
    prev = None
    for s in SCALES:
        r = solve(torus_kernel(s), Rt, TORUS["gamma"], TORUS["kappa"], start=prev, omega=0.5)
        prev = np.array(r.mass, copy=True)
        name = f"decay_torus_s{s:g}"
        d = r.mass / Rt
        summary["torus"][name] = {"scale": s, "status": r.status.name, "iterations": r.iterations,
                                  "max_density": float(d.max()), "centers": torus_centers(d), **margin(r)}
        report(name, r, Rt)
        save(name, r)
    Rr = ring_capacity()
    print("Figure 4, ring (gamma=0.5, kappa=0.0128):", flush=True)
    prev = None
    for s in SCALES:
        r = solve(ring_kernel(s), Rr, RING["gamma"], RING["kappa"], start=prev, omega=0.5)
        prev = np.array(r.mass, copy=True)
        name = f"decay_ring_s{s:g}"
        d = r.mass / Rr
        summary["ring"][name] = {"scale": s, "status": r.status.name, "iterations": r.iterations,
                                 "max_density": float(d.max()), "peaks": len(ring_peaks(d)), **margin(r)}
        report(name, r, Rr)
        save(name, r)
    with open(f"{ARCHIVE}/fig4_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
