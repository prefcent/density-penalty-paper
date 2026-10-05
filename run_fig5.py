"""Figure 5: one landscape, two equilibria.

Ring, gamma = 1.75, kappa = 0.0128, omega = 0.5, identical inputs. Run A starts from
the capacities; run B starts from the capacities with a spike added at zone 478,
the second of two nearly equal capacity hills (zones 450 and 479), renormalized to
the same total. Each converged state is then perturbed multiplicatively by
1 + 1e-4 * N(0, 1) (seed 13), renormalized, and re-solved; the relative distance
between the re-solved state and the original is recorded.
"""
import json

import numpy as np

from common import ARCHIVE, RING_N, ring_capacity, ring_kernel, ring_peaks, margin, report, save, solve

GAMMA, KAPPA = 1.75, 0.0128
SEED_ZONE, SEED_HEIGHT = 478, 0.2 * 50   # spike of 10 mean-capacity units at zone 478
PERTURB_SEED, PERTURB_SIZE = 13, 1e-4


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))


def main() -> None:
    R = ring_capacity()
    K = ring_kernel()
    seeded = np.array(R, copy=True)
    seeded[SEED_ZONE] += SEED_HEIGHT * R.sum() / RING_N
    seeded *= R.sum() / seeded.sum()
    print("Figure 5 (ring, gamma=1.75, kappa=0.0128):", flush=True)
    runs = {"siting_A": solve(K, R, GAMMA, KAPPA, omega=0.5),
            "siting_B": solve(K, R, GAMMA, KAPPA, start=seeded, omega=0.5)}
    summary = {"seed_zone": SEED_ZONE, "perturbation": {"seed": PERTURB_SEED, "relative_size": PERTURB_SIZE},
               "distance_A_B": rel(runs["siting_A"].mass, runs["siting_B"].mass), "runs": {}}
    for name, r in runs.items():
        rng = np.random.default_rng(PERTURB_SEED)
        pert = r.mass * (1.0 + PERTURB_SIZE * rng.standard_normal(RING_N))
        pert *= R.sum() / pert.sum()
        back = solve(K, R, GAMMA, KAPPA, start=pert, omega=0.5)
        d = r.mass / R
        peaks = ring_peaks(d)
        summary["runs"][name] = {"status": r.status.name, "iterations": r.iterations,
                                 "final_step_norm": r.final_step_norm,
                                 "start_convention": r.manifest.fingerprint["solver"]["start_convention"],
                                 "peaks": [int(i) for i in peaks],
                                 "peak_densities": [float(d[i]) for i in peaks],
                                 "perturbation_return_distance": rel(back.mass, r.mass), **margin(r)}
        report(name, r, R)
        print(f"    peaks {list(map(int, peaks))}; perturbation returns to within "
              f"{summary['runs'][name]['perturbation_return_distance']:.1e}", flush=True)
        save(name, r)
    fa, fb = runs["siting_A"].manifest.fingerprint, runs["siting_B"].manifest.fingerprint
    summary["fingerprints_identical_except_solver_fields"] = [
        k for k in fa["solver"] if fa["solver"][k] != fb["solver"][k]]
    summary["kernel_landscape_model_identical"] = all(fa[k] == fb[k] for k in ("kernel", "landscape", "model"))
    print(f"  A vs B relative distance {summary['distance_A_B']:.2f}; solver fields differing: "
          f"{summary['fingerprints_identical_except_solver_fields']}", flush=True)
    with open(f"{ARCHIVE}/fig5_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
