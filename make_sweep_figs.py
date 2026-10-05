"""Render Figures 6-8 and the Table 2 numbers from archive/sweeps (no solving here).

Figure 6: distinct converged states over (gamma, s), sweep grid_s.
Figure 7: distinct converged states over (gamma, sigma), sweep grid_sigma, with the patterns.
Figure 8: one grid point of grid_s whose states differ in their number of centers.
figs/table2_sweeps.txt: one line per sweep, plus the counting diagnostics quoted in the text.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import sweeps as sw
from common import FIGS, TORUS_N, torus_centers

INK, MUTED = "#1a1a1a", "#6b6b6b"
# 1 state = neutral; 2, 3, 4, 5, 6+ = one-hue ramp, light -> dark
CMAP = ListedColormap(["#efeeea", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 99], CMAP.N)
plt.rcParams.update({"font.size": 8, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.dpi": 200})
LANDS = {p: i + 1 for i, p in enumerate(sw.RANDOM)}


def summary(sweep: str) -> list[dict]:
    with open(sw.ARCHIVE / f"{sweep}.summary.json") as fh:
        return json.load(fh)["points"]


def savefig(fig, stem: str) -> None:
    fig.savefig(f"{FIGS}/{stem}.png", bbox_inches="tight")
    fig.savefig(f"{FIGS}/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"{stem} written", flush=True)


def differ(p: dict) -> bool:
    return p["n_found"] > 1 and len({st["centers"] for st in p["states"]}) > 1


def count_map(ax, pts: list[list[dict]], xlabels: list[str], ylabels: list[str] | None,
              outline: bool) -> None:
    n = np.array([[p["n_found"] for p in row] for row in pts])
    ax.imshow(n, cmap=CMAP, norm=NORM, origin="lower", aspect="auto")
    for i, row in enumerate(pts):
        for j, p in enumerate(row):
            mark = outline and differ(p)
            ax.text(j, i, f"{p['n_found']}{'†' if p['n_budget'] else ''}", ha="center", va="center",
                    fontsize=7, color="white" if p["n_found"] >= 4 else INK,
                    fontweight="bold" if mark else "normal")
            if mark:
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                           edgecolor=INK, linewidth=1.0))
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=6.5)
    ax.set_yticks(range(len(pts)), ylabels if ylabels else [])
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pts)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig6() -> None:
    g, idx = sw.SWEEPS["grid_s"], {}
    for p in summary("grid_s"):
        idx[(p["land"], p["gamma"], p["s"])] = p
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    for r, ls in enumerate((0.1, 0.5)):
        for c, pat in enumerate(sw.RANDOM):
            ax = axes[r, c]
            pts = [[idx[(sw.land_key(pat, ls), gam, s)] for s in g["scales"]] for gam in g["gammas"]]
            count_map(ax, pts, [f"{s:g}" for s in g["scales"]],
                      [f"{gam:g}" for gam in g["gammas"]] if c == 0 else None, outline=True)
            if r == 0:
                ax.set_title(f"landscape {c + 1}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"$\\sigma = {ls:g}$\n\n$\\gamma$")
            if r == 1:
                ax.set_xlabel("line-haul cost factor $s$")
    fig.tight_layout()
    savefig(fig, "fig6_multistart")


def fig7() -> None:
    g, idx = sw.SWEEPS["grid_sigma"], {}
    for p in summary("grid_sigma"):
        idx[(p["land"], p["kappa"], p["gamma"])] = p
    sigmas = sorted({ls for _, ls in g["lands"]})
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62, 1, 1], hspace=0.32, wspace=0.12)
    for c, pat in enumerate(sw.RANDOM):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(np.log(sw.capacity(sw.land_key(pat, 0.5))).reshape(TORUS_N, TORUS_N) / 0.5,
                  cmap="Greys", vmin=-3, vmax=3, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"landscape {c + 1}: pattern", fontsize=8)
    for r, kap in enumerate(g["kappas"]):
        for c, pat in enumerate(sw.RANDOM):
            ax = fig.add_subplot(gs[r + 1, c])
            pts = [[idx[(sw.land_key(pat, ls), kap, gam)] for ls in sigmas] for gam in g["gammas"]]
            count_map(ax, pts, [f"{ls:g}" for ls in sigmas],
                      [f"{gam:g}" for gam in g["gammas"]] if c == 0 else None, outline=False)
            if c == 0:
                ax.set_ylabel(f"$\\kappa = {kap:g}$\n\n$\\gamma$")
            if r == len(g["kappas"]) - 1:
                ax.set_xlabel("capacity log-std $\\sigma$")
    savefig(fig, "fig7_amplitude_map")


def pick_pair(pts: list[dict]) -> dict:
    """Two states with different center counts, each reached from >= 3 starts; prefer the
    most evenly visited pair, then s nearest 1, then the smallest gamma."""
    cands = [p for p in pts if p["n_found"] == 2 and differ(p)
             and min(len(st["starts"]) for st in p["states"]) >= 3]
    return min(cands, key=lambda p: (-min(len(st["starts"]) for st in p["states"]),
                                     abs(np.log(p["s"])), p["gamma"], p["land"]))


def fig8() -> dict:
    p = pick_pair(summary("grid_s"))
    states = np.load(sw.ARCHIVE / "grid_s.states.npz")
    R = sw.capacity(p["land"])
    ds = [states[st["rep"]] / R for st in p["states"]]
    vmax = max(float(d.max()) for d in ds)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"wspace": 0.25})
    for ax, d, st in zip(axes, ds, p["states"]):
        im = ax.imshow(np.log2(d.reshape(TORUS_N, TORUS_N)), cmap="Blues", vmin=0,
                       vmax=np.log2(vmax), interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{torus_centers(d)} centers; reached from {len(st['starts'])} of "
                     f"{p['starts']} starts\nmax density {st['max_density']:.1f}", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("$\\log_2$ density $a/R$ (values < 0 shown at 0)", fontsize=7)
    savefig(fig, "fig8_state_pair")
    return p


def table(pair: dict) -> None:
    lines = [f"{'sweep':12s} {'points':>6s} {'>=2':>5s} {'differ':>6s} {'max':>4s} {'budget':>6s} "
             f"{'domain':>6s} {'states':>6s} {'returned':>8s}"]
    within, between, ret = 0.0, np.inf, 0.0
    for sweep in sw.SWEEPS:
        pts = summary(sweep)
        lines.append(f"{sweep:12s} {len(pts):6d} {sum(p['n_found'] > 1 for p in pts):5d} "
                     f"{sum(differ(p) for p in pts):6d} {max(p['n_found'] for p in pts):4d} "
                     f"{sum(p['n_budget'] for p in pts):6d} {sum(p['n_domain'] for p in pts):6d} "
                     f"{sum(p['n_found'] for p in pts if p['n_found'] > 1):6d} "
                     f"{sum(p['n_returned'] for p in pts if p['n_found'] > 1):8d}")
        if sweep in ("grid_s", "grid_sigma"):
            within = max(within, max(p["max_within"] for p in pts))
            between = min(between, min(p["min_between"] for p in pts if p["min_between"] is not None))
            ret = max(ret, max(st["return"]["return_dist"] for p in pts for st in p["states"]))
    lines += ["", "grid_s and grid_sigma:",
              f"  largest distance within one state:   {within:.2e}",
              f"  smallest distance between states:    {between:.2f}",
              f"  largest perturbation-return distance: {ret:.2e}",
              f"  Figure 8 point: {pair['land']} gamma={pair['gamma']:g} kappa={pair['kappa']:g} s={pair['s']:g}"]
    with open(f"{FIGS}/table2_sweeps.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    fig6()
    fig7()
    table(fig8())
