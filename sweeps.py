"""Multi-start sweeps on the 64 x 64 torus (Figures 6-8, Appendix A, Table 2).

At every grid point the model is run from the same fixed-seed initial states, and the
distinct converged states are counted. Each distinct state then gets a perturbation-return
test. The protocol, as stated in the note's appendix:

* landscapes: a fixed correlated pattern per seed (periodic Gaussian smoothing, width
  4 cells), standardized, capacity proportional to exp(sigma * z), mean 1; plus the
  four-hill landscape of Figure 3 in the preliminary grids;
* initial states: a = R; uniform; R * exp(w) with w a correlated field (width 3 cells,
  unit std); R * (1 + 4 * sum of 2-5 Gaussian bumps of width 3 cells at random sites);
  all rescaled to the total capacity; the same states at every grid point of a landscape;
* runs: omega = 0.5, tolerance 1e-12 on the normalized step, budget 400,000 iterations,
  torus kernel applied by FFT (torus_fft.py), one BLAS thread;
* counting: converged states are grouped greedily -- a state joins the first earlier group
  whose first state lies within SAME_TOL in the normalized max norm; budget-exhausted and
  domain-stopped starts are recorded and not counted;
* perturbation-return: each distinct state is multiplied by (1 + 1e-4 * xi), xi standard
  normal with a fixed seed, renormalized, and run to tolerance again.

Usage:
    python3 sweeps.py run [SWEEP ...]     solve (resumable, in work/), analyze, pack archive/sweeps/
    python3 sweeps.py pack [SWEEP ...]    re-analyze finished runs and repack
The sweep names are the keys of SWEEPS.
"""
from __future__ import annotations

import os

# One BLAS thread per worker, set before numpy loads; recorded in every manifest.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import functools
import gzip
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import prefcent as pc
from common import (BETA, CELL_M, D0, RHO, TORUS_N, _smooth_periodic, hills_capacity,
                    torus_centers)
from torus_fft import torus_fft_kernel

ARCHIVE = Path("archive/sweeps")
WORK = Path(os.environ.get("SWEEP_WORK", "work"))

OMEGA, TOL, MAX_ITER = 0.5, 1e-12, 400_000
SAME_TOL = 1e-6
PERTURB = 1e-4
PATTERN_SEEDS = (101, 202, 303)
PATTERN_WIDTH = 4.0      # cells (km)

G7 = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
G10 = G7 + (2.5, 3.0, 4.0)
S7 = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
RANDOM = tuple(f"rand{s}" for s in PATTERN_SEEDS)

# name -> grid. "lands" are (pattern, sigma); sigma is None for the four-hill landscape.
SWEEPS: dict[str, dict] = {
    **{f"prelim_k{k:g}": dict(lands=[(p, 0.5) for p in RANDOM] + [("hills", None)],
                              gammas=G7, scales=S7, kappas=(k,), starts=tuple(range(8)))
       for k in (0.05, 0.02, 0.01)},
    "amplitude": dict(lands=[(p, ls) for p in RANDOM for ls in (0.01, 0.02, 0.05, 0.1, 0.2)],
                      gammas=G7, scales=(1.0,), kappas=(0.02,), starts=tuple(range(12))),
    "probe": dict(lands=[(p, ls) for p in RANDOM for ls in (0.01, 0.05)],
                  gammas=(2.5, 3.0, 4.0), scales=(1.0, 3.0), kappas=(0.01, 0.005),
                  starts=(0, 2, 5, 8)),
    "grid_sigma": dict(lands=[(p, ls) for p in RANDOM for ls in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)],
                       gammas=G10, scales=(1.0,), kappas=(0.01, 0.005), starts=tuple(range(16))),
    "grid_s": dict(lands=[(p, ls) for p in RANDOM for ls in (0.1, 0.5)],
                   gammas=G10, scales=S7, kappas=(0.01,), starts=tuple(range(16))),
}


# --------------------------------------------------------------- landscapes ---
def land_key(pattern: str, sigma: float | None) -> str:
    return pattern if sigma is None else f"{pattern}ls{sigma:g}"


@functools.lru_cache(maxsize=64)
def capacity(key: str) -> np.ndarray:
    if key == "hills":
        return hills_capacity()
    m = re.fullmatch(r"rand(\d+)ls([0-9.]+)", key)
    seed, sigma = int(m[1]), float(m[2])
    z = _smooth_periodic(np.random.default_rng(seed).standard_normal((TORUS_N, TORUS_N)), PATTERN_WIDTH)
    r = np.exp(z / z.std() * sigma).reshape(-1)
    return r * (r.size / r.sum())


def pattern_of(key: str) -> str:
    return re.sub(r"ls[0-9.]+$", "", key)


def start_state(key: str, k: int) -> np.ndarray | None:
    """Initial state k; depends on the landscape's pattern only, not on sigma or the grid point."""
    R = capacity(key)
    if k == 0:
        return None                      # prefcent's default start, a = R
    total = R.sum()
    if k == 1:
        return np.full_like(R, total / R.size)
    rng = np.random.default_rng([sum(map(ord, pattern_of(key))), 1000 + k])
    if k in (2, 3, 4, 8, 9, 12, 13):
        w = _smooth_periodic(rng.standard_normal((TORUS_N, TORUS_N)), 3.0)
        a = R * np.exp(w / w.std()).reshape(-1)
    else:
        n = TORUS_N
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        bumps = np.zeros((n, n))
        for _ in range(int(rng.integers(2, 6))):
            cy, cx = rng.integers(0, n, size=2)
            dy = np.minimum(np.abs(yy - cy), n - np.abs(yy - cy))
            dx = np.minimum(np.abs(xx - cx), n - np.abs(xx - cx))
            bumps += np.exp(-(dy ** 2 + dx ** 2) / (2 * 3.0 ** 2))
        a = R * (1.0 + 4.0 * bumps.reshape(-1))
    return a * (total / a.sum())


# --------------------------------------------------------------------- runs ---
@functools.lru_cache(maxsize=16)
def kernel(scale: float) -> pc.OperatorKernel:
    return torus_fft_kernel(TORUS_N, CELL_M, scale, D0, BETA)


def model(key: str, gamma: float, scale: float, kappa: float) -> pc.Model:
    return pc.Model(kernel(scale), pc.Landscape(capacity(key)), gamma=gamma,
                    closure=pc.DensityPenaltyV1(kappa, RHO))


def run_name(key: str, gamma: float, scale: float, kappa: float, k: int) -> str:
    return f"{key}_k{kappa:g}_g{gamma:g}_s{scale:g}_st{k}"


def jobs(sweep: str) -> list[tuple]:
    g = SWEEPS[sweep]
    return [(land_key(p, ls), gam, s, kap, k) for (p, ls) in g["lands"] for kap in g["kappas"]
            for s in g["scales"] for gam in g["gammas"] for k in g["starts"]]


def solve_one(sweep: str, key: str, gamma: float, scale: float, kappa: float, k: int) -> dict:
    """Solve one start; resumable (a finished run's record is reused)."""
    d = WORK / sweep
    name = run_name(key, gamma, scale, kappa, k)
    rec_path = d / f"{name}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    rec: dict = {"run": name, "land": key, "gamma": gamma, "s": scale, "kappa": kappa, "start": k}
    t0 = time.time()
    try:
        r = model(key, gamma, scale, kappa).evolve(start=start_state(key, k), max_iter=MAX_ITER,
                                                   tol=TOL, omega=OMEGA)
        cert = r.certificates["closure:density_penalty_v1"]
        rec.update(status=r.status.name, iterations=r.iterations, final_step=float(r.final_step_norm),
                   max_density=float((r.mass / capacity(key)).max()), margin_passed=bool(cert.passed),
                   kappa_star=cert.data["kappa_at_zero"], manifest=json.loads(r.manifest.to_json()))
        np.save(d / f"{name}.mass.npy", r.mass)
    except pc.PrefcentError as e:        # domain stops are outcomes, recorded and not counted
        rec.update(status="DOMAIN_STOP", error=type(e).__name__, message=str(e)[:300])
    rec["wall_s"] = round(time.time() - t0, 2)
    rec_path.write_text(json.dumps(rec))
    return rec


# ----------------------------------------------------------------- counting ---
def dist(a: np.ndarray, b: np.ndarray, key: str) -> float:
    """Normalized max norm of the stopping rule: max |a - b| / mean(R)."""
    return float(np.max(np.abs(a - b)) / capacity(key).mean())


def shift_dist(a: np.ndarray, b: np.ndarray, key: str) -> float:
    """Smallest max-norm difference of the density patterns over all torus shifts."""
    R = capacity(key)
    n = TORUS_N
    da, db = (a / R).reshape(n, n), (b / R).reshape(n, n)
    best = np.inf
    for dy in range(n):
        rolled = np.roll(db, dy, axis=0)
        for dx in range(n):
            best = min(best, float(np.max(np.abs(da - np.roll(rolled, dx, axis=1)))))
    return best


def perturb_return(sweep: str, key: str, gamma: float, scale: float, kappa: float, rep: str, idx: int) -> dict:
    a = np.load(WORK / sweep / f"{rep}.mass.npy")
    rng = np.random.default_rng([sum(map(ord, key)), int(gamma * 100), int(scale * 100),
                                 int(kappa * 1e4), idx, 7])
    p = a * (1.0 + PERTURB * rng.standard_normal(a.size))
    p *= a.sum() / p.sum()
    try:
        r = model(key, gamma, scale, kappa).evolve(start=p, max_iter=MAX_ITER, tol=TOL, omega=OMEGA)
        return {"status": r.status.name, "iterations": r.iterations, "return_dist": dist(r.mass, a, key)}
    except pc.PrefcentError as e:
        return {"status": "DOMAIN_STOP", "error": type(e).__name__}


def group(sweep: str, recs: list[dict]) -> dict:
    r0 = recs[0]
    key = r0["land"]
    states: list[tuple[str, np.ndarray, list[int]]] = []
    max_within = 0.0
    for r in sorted((r for r in recs if r["status"] == "CONVERGED"), key=lambda r: r["start"]):
        a = np.load(WORK / sweep / f"{r['run']}.mass.npy")
        for st in states:
            dd = dist(a, st[1], key)
            if dd <= SAME_TOL:
                st[2].append(r["start"])
                max_within = max(max_within, dd)
                break
        else:
            states.append((r["run"], a, [r["start"]]))
    pairs = [{"i": i, "j": j, "dist": dist(states[i][1], states[j][1], key),
              "shift_dist": shift_dist(states[i][1], states[j][1], key)}
             for i in range(len(states)) for j in range(i)]
    R = capacity(key)
    return {
        "land": key, "gamma": r0["gamma"], "s": r0["s"], "kappa": r0["kappa"],
        "starts": len(recs), "n_found": len(states),
        "n_budget": sum(r["status"] == "BUDGET_EXHAUSTED" for r in recs),
        "n_domain": sum(r["status"] == "DOMAIN_STOP" for r in recs),
        "max_within": max_within, "min_between": min((q["dist"] for q in pairs), default=None),
        "pairs": pairs,
        "states": [{"rep": n, "starts": st, "max_density": float((a / R).max()),
                    "centers": torus_centers(a / R)} for n, a, st in states],
    }


def pack(sweep: str, workers: int) -> None:
    recs = [json.loads(p.read_text()) for p in sorted((WORK / sweep).glob("*_st*.json"))]
    want = {run_name(*j) for j in jobs(sweep)}
    have = {r["run"] for r in recs}
    if want - have:
        raise SystemExit(f"{sweep}: {len(want - have)} runs missing; run first")
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        if r["run"] in want:
            groups.setdefault((r["land"], r["kappa"], r["s"], r["gamma"]), []).append(r)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        points = list(ex.map(group, [sweep] * len(groups), list(groups.values()), chunksize=4))
        tests = [(sweep, p["land"], p["gamma"], p["s"], p["kappa"], st["rep"], i)
                 for p in points for i, st in enumerate(p["states"])]
        results = list(ex.map(perturb_return, *zip(*tests), chunksize=4)) if tests else []
    by_rep = {t[5]: r for t, r in zip(tests, results)}
    for p in points:
        for st in p["states"]:
            st["return"] = by_rep[st["rep"]]
        p["n_returned"] = sum(st["return"].get("return_dist", np.inf) <= SAME_TOL for st in p["states"])
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with gzip.open(ARCHIVE / f"{sweep}.runs.jsonl.gz", "wt") as fh:
        for r in sorted((r for r in recs if r["run"] in want), key=lambda r: r["run"]):
            fh.write(json.dumps({k: v for k, v in r.items() if k != "wall_s"}, sort_keys=True) + "\n")
    np.savez_compressed(ARCHIVE / f"{sweep}.states.npz",
                        **{st["rep"]: np.load(WORK / sweep / f"{st['rep']}.mass.npy")
                           for p in points for st in p["states"]})
    grid = {k: v for k, v in SWEEPS[sweep].items()}
    summary = {"sweep": sweep, "grid": grid, "protocol": {
        "omega": OMEGA, "tol": TOL, "max_iter": MAX_ITER, "same_tol": SAME_TOL, "perturb": PERTURB,
        "norm": "max|a-b|/mean(R)", "centers": "9x9 periodic-neighborhood density maxima above 1.3"},
        "points": points}
    (ARCHIVE / f"{sweep}.summary.json").write_text(json.dumps(summary, indent=1))
    multi = sum(p["n_found"] > 1 for p in points)
    print(f"{sweep}: {len(points)} points, {multi} with >= 2 distinct states, "
          f"{sum(p['n_budget'] for p in points)} budget / {sum(p['n_domain'] for p in points)} domain "
          f"starts not counted", flush=True)


def default_workers() -> int:
    """Pool size from the cgroup CPU quota when there is one (os.cpu_count() can report the host)."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)) - 4)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) - 1)


def run(sweep: str, workers: int) -> None:
    (WORK / sweep).mkdir(parents=True, exist_ok=True)
    js = jobs(sweep)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # group by scale so each worker reuses its cached kernel
        recs = list(ex.map(solve_one, *zip(*[(sweep, *j) for j in sorted(js, key=lambda j: (j[2], j))]),
                           chunksize=8))
    status: dict[str, int] = {}
    for r in recs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    print(f"{sweep}: {len(js)} runs {status} in {time.time() - t0:.0f} s", flush=True)
    pack(sweep, workers)


def main() -> None:
    cmd, *names = sys.argv[1:] or ["run"]
    names = names or list(SWEEPS)
    workers = int(os.environ.get("SWEEP_WORKERS", default_workers()))
    for name in names:
        (run if cmd == "run" else pack)(name, workers)


if __name__ == "__main__":
    main()
