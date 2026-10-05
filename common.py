"""Shared inputs for the preprint's synthetic experiments.

Two landscapes, two kernels, and the fixed peak/center rule. Every figure script
imports from here, so the seeds and the geometry are defined in exactly one place.

Units: costs are metre-equivalents at 60 km/h (1000 per zone at 1 km apart), and
``D0 = 5000`` is a five-minute terminal penalty added to every trip.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

import prefcent as pc

ARCHIVE = "archive"
FIGS = "figs"

BETA, D0, RHO = 2.0, 5000.0, 2.0
CELL_M = 1000.0            # zones are 1 km apart on both the ring and the torus
RING_N = 512
TORUS_N = 64
SEED = 42                  # the correlated ring landscape
LOG_STD = 0.5              # log-std of the ring's lognormal capacity
RING_CORR_KM = 10.0        # correlation length of the ring landscape


# ------------------------------------------------------------------ kernels ---
def ring_kernel(scale: float = 1.0) -> pc.CirculantKernel:
    """The 512-zone ring, line-haul costs scaled by ``scale``, terminal penalty fixed."""
    return pc.CirculantKernel.ring(RING_N, CELL_M * scale, D0, BETA, self_interaction=False)


def torus_costs(n_side: int = TORUS_N, cell_m: float = CELL_M) -> np.ndarray:
    """Euclidean costs on an n x n torus with periodic wrap, as a dense matrix."""
    ij = np.arange(n_side)
    dx = np.minimum(np.abs(ij[:, None] - ij[None, :]), n_side - np.abs(ij[:, None] - ij[None, :])) * cell_m
    d = (dx[:, None, :, None] ** 2 + dx[None, :, None, :] ** 2) ** 0.5
    return np.ascontiguousarray(d.reshape(n_side * n_side, n_side * n_side))


def torus_kernel(scale: float = 1.0) -> pc.DenseKernel:
    """The 64 x 64 torus, line-haul costs scaled by ``scale``, terminal penalty fixed."""
    units = "metre-equivalents" if scale == 1.0 else f"metre-equivalents (line-haul scaled by {scale:g})"
    return pc.kernels.from_costs(torus_costs() * scale, beta=BETA, d0=D0,
                                 cost_units=units, self_interaction=False)


# --------------------------------------------------------------- landscapes ---
def _smooth_periodic(field: np.ndarray, corr_len: float) -> np.ndarray:
    """Gaussian smoothing with periodic wrap (any dimensionality), via FFT."""
    f = np.fft.fftn(field)
    for ax, n in enumerate(field.shape):
        q = np.fft.fftfreq(n)
        shape = [1] * field.ndim
        shape[ax] = n
        f = f * np.exp(-2 * (np.pi * q.reshape(shape) * corr_len) ** 2)
    return np.real(np.fft.ifftn(f))


def ring_capacity() -> np.ndarray:
    """Spatially correlated lognormal capacity on the ring: a smoothed Gaussian
    log-field rescaled to log-std ``LOG_STD``, exponentiated, scaled to mean 1."""
    rng = np.random.default_rng(SEED)
    g = _smooth_periodic(rng.standard_normal((RING_N,)), RING_CORR_KM)
    g = g / g.std() * LOG_STD
    r = np.exp(g).reshape(-1)
    return r * (r.size / r.sum())


def hills_capacity(n: int = TORUS_N) -> np.ndarray:
    """Four designed capacity hills on the torus (Gaussian bumps of decreasing
    height, roughly 30 km apart) over a weak correlated noise floor; mean 1."""
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    def bump(cy, cx, amp, w):
        dy = np.minimum(np.abs(yy - cy), n - np.abs(yy - cy))
        dx = np.minimum(np.abs(xx - cx), n - np.abs(xx - cx))
        return amp * np.exp(-(dy ** 2 + dx ** 2) / (2 * w ** 2))

    field = (bump(14, 14, 1.6, 5) + bump(14, 46, 1.2, 5)
             + bump(46, 20, 1.0, 5) + bump(44, 48, 0.8, 5))
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((n, n))
    f = np.fft.fft2(noise)
    q = np.fft.fftfreq(n)
    f *= np.exp(-2 * (np.pi * 3.0) ** 2 * (q[:, None] ** 2 + q[None, :] ** 2))
    noise = np.real(np.fft.ifft2(f))
    noise *= 0.15 / noise.std()
    R = np.exp(0.8 * field + noise)      # the tallest hill is about 3x the mean
    R = R.reshape(-1)
    return R * (R.size / R.sum())


# ------------------------------------------------------------ the fixed rule ---
def ring_peaks(density: np.ndarray, level: float = 1.3) -> np.ndarray:
    """Indices of strict local density maxima above ``level`` on the ring."""
    d = density
    return np.flatnonzero((d > np.roll(d, 1)) & (d > np.roll(d, -1)) & (d > level))


def torus_centers(density: np.ndarray, level: float = 1.3, size: int = 9) -> int:
    """Number of ``size`` x ``size``-neighborhood density maxima above ``level``.

    Neighborhoods wrap around the torus edges, so a center that crosses an edge
    counts once.
    """
    d = density.reshape(TORUS_N, TORUS_N)
    return int(((d == maximum_filter(d, size=size, mode="wrap")) & (d > level)).sum())


# ------------------------------------------------------------------ helpers ---
def solve(kernel, capacity: np.ndarray, gamma: float, kappa: float, *, start=None,
          omega: float = 1.0, tol: float | None = 1e-12, max_iter: int = 400_000) -> pc.EvolveResult:
    """One run of the penalized model. ``tol=None`` is a fixed-budget run."""
    model = pc.Model(kernel, pc.Landscape(capacity), gamma=gamma,
                     closure=pc.DensityPenaltyV1(kappa, RHO))
    return model.evolve(start=start, max_iter=max_iter, tol=tol, omega=omega)


def save(name: str, result: pc.EvolveResult) -> None:
    np.save(f"{ARCHIVE}/{name}.mass.npy", result.mass)
    with open(f"{ARCHIVE}/{name}.manifest.json", "w") as fh:
        fh.write(result.manifest.to_json())


def margin(result: pc.EvolveResult) -> dict:
    cert = result.certificates["closure:density_penalty_v1"]
    return {"margin_passed": bool(cert.passed),
            "kappa_at_zero": cert.data["kappa_at_zero"],
            "min_closed_mass": float(cert.data["min_closed_mass"])}


def report(name: str, result: pc.EvolveResult, capacity: np.ndarray) -> None:
    d = result.mass / capacity
    m = margin(result)
    print(f"  {name:28s} {result.status.name:13s} it={result.iterations:<6d} "
          f"step={result.final_step_norm:.3e} max density={float(d.max()):8.3f} "
          f"margin={m['margin_passed']} kappa*={m['kappa_at_zero']}", flush=True)
