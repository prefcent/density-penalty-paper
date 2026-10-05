"""Figure 2: the penalty as a dispersal dial.

Ring, gamma = 0.5. The unpenalized equilibrium is solved first (omega = 1, start
a = R); its margin certificate gives kappa*, and the series kappa = 10 %, 25 %, 50 %
of that value is solved by continuation in kappa (warm start from the previous
equilibrium, omega = 0.5). Every run must meet tolerance 1e-12 and pass its margin
certificate.
"""
import json

import numpy as np

from common import ARCHIVE, ring_capacity, ring_kernel, ring_peaks, margin, report, save, solve

GAMMA = 0.5
FRACTIONS = (0.10, 0.25, 0.50)


def main() -> None:
    R = ring_capacity()
    K = ring_kernel()
    print("Figure 2 (ring, gamma=0.5):", flush=True)
    r0 = solve(K, R, GAMMA, 0.0, omega=1.0)
    kappa_star = float(margin(r0)["kappa_at_zero"])
    series = [round(f * kappa_star, 4) for f in FRACTIONS]
    summary = {"kappa_star_unpenalized": kappa_star, "series": series, "runs": {}}
    prev = r0
    for kappa in [0.0] + series:
        r = r0 if kappa == 0.0 else solve(K, R, GAMMA, kappa, start=np.array(prev.mass, copy=True), omega=0.5)
        prev = r
        name = f"ring512_kappa{kappa:g}"
        d = r.mass / R
        summary["runs"][name] = {"kappa": kappa, "status": r.status.name, "iterations": r.iterations,
                                 "final_step_norm": r.final_step_norm, "max_density": float(d.max()),
                                 "peaks": len(ring_peaks(d)), **margin(r)}
        report(name, r, R)
        save(name, r)
    with open(f"{ARCHIVE}/fig2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
