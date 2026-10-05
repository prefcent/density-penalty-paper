"""Render Figures 1-5 from the archived states and manifests (no solving here)."""
from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from common import ARCHIVE, FIGS, TORUS_N, hills_capacity, ring_capacity, ring_peaks, torus_centers

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e3"
BLUES = ["#b8d0e8", "#7faed6", "#3f7fb8", "#123e66"]   # one-hue ramp for ordered series
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "text.color": INK, "figure.dpi": 200,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def state(name: str) -> np.ndarray:
    return np.load(f"{ARCHIVE}/{name}.mass.npy")


def manifest(name: str) -> dict:
    with open(f"{ARCHIVE}/{name}.manifest.json") as fh:
        return json.load(fh)


def savefig(fig, stem: str) -> None:
    fig.savefig(f"{FIGS}/{stem}.png", bbox_inches="tight")
    fig.savefig(f"{FIGS}/{stem}.pdf", bbox_inches="tight")
    print(f"{stem} written", flush=True)


R_ring = ring_capacity()
x = np.arange(R_ring.size)

# ---------------- Fig 1: the 200-iteration budget pair (gamma=2, omega=1) ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, kappa, color in zip(axes, (0.0, 0.0128), (BLUES[3], BLUES[2])):
    name = f"fig1_ring_g2_k{kappa:g}_b200"
    obs = manifest(name)["observations"]
    d = state(name) / R_ring
    ax.plot(x, d, lw=1.6, color=color, zorder=3)
    style(ax)
    ax.set_yscale("log")
    ax.set_xlabel("zone")
    ax.set_title(f"κ = {kappa:g}\n{obs['status']} after {obs['iterations']} iterations",
                 fontsize=9.5, color=INK)
axes[0].set_ylabel("density  a/R")
fig.tight_layout()
savefig(fig, "fig1_rescue")

# ---------------- Fig 2: ring kappa series (gamma=0.5) ----------------
kappas = [0.0] + json.load(open(f"{ARCHIVE}/fig2_summary.json"))["series"]
fig = plt.figure(figsize=(7.0, 3.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.12)
axR = fig.add_subplot(gs[0])
axD = fig.add_subplot(gs[1], sharex=axR)
axR.fill_between(x, R_ring, color="#d9d9d9", lw=0, zorder=2)
axR.set_ylabel("capacity R", fontsize=9)
axR.tick_params(labelbottom=False)
style(axR)
for k, c in zip(kappas, BLUES[::-1]):
    d = state(f"ring512_kappa{k:g}") / R_ring
    axD.plot(x, d, lw=1.7, color=c, zorder=3, label=f"κ = {k:g}")
style(axD)
axD.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, handlelength=1.6, borderaxespad=0.2)
axD.set_xlabel("zone")
axD.set_ylabel("density  a/R")
axD.set_xlim(0, x[-1])
savefig(fig, "fig2_ring_series")

# ---------------- Fig 3: four-hill torus, two penalties ----------------
Rt = hills_capacity().reshape(TORUS_N, TORUS_N)
dA = state("fig3_hills_g1.5_k0.1").reshape(TORUS_N, TORUS_N) / Rt
dB = state("fig3_hills_g1.5_k0.02").reshape(TORUS_N, TORUS_N) / Rt
lA, lB = np.log2(dA), np.log2(dB)
lim = float(max(abs(lA).max(), abs(lB).max()))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3), constrained_layout=True)
axes[0].imshow(Rt, cmap="Greys", origin="lower")
axes[0].set_title("capacity R", fontsize=9, color=INK)
for ax, l, d, kappa in zip(axes[1:], (lA, lB), (dA, dB), (0.1, 0.02)):
    im = ax.imshow(l, cmap="RdBu", norm=norm, origin="lower")
    ax.set_title(f"κ = {kappa:g}   (max density {float(d.max()):.1f})", fontsize=9, color=INK)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
cbar = fig.colorbar(im, ax=axes[1:], location="bottom", shrink=0.6, pad=0.02, aspect=35)
cbar.set_label("log₂ density  (blue: above capacity share; red: below; shared scale)", fontsize=9)
cbar.outline.set_edgecolor(MUTED)
savefig(fig, "fig3_torus_pair")

# ---------------- Fig 4: transport cost vs number of centers ----------------
S_VALUES = (10.0, 3.0, 1.0, 0.3)
S_LABELS = ("line-haul ×10", "line-haul ×3", "baseline ×1", "line-haul ×0.3")
fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.6), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.25, 1.0]})
lim = 2.4
for ax, sv, lab in zip(axes[0], S_VALUES, S_LABELS):
    d = state(f"decay_torus_s{sv:g}") / Rt.reshape(-1)
    im = ax.imshow(np.log2(d.reshape(TORUS_N, TORUS_N)), cmap="RdBu", norm=TwoSlopeNorm(0.0, -lim, lim), origin="lower")
    ax.set_title(f"{lab}\ncenters: {torus_centers(d)}   max density {float(d.max()):.1f}", fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
cbar = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.015)
cbar.set_label("log₂ density", fontsize=10); cbar.outline.set_edgecolor(MUTED)
for ax, sv in zip(axes[1], S_VALUES):
    d = state(f"decay_ring_s{sv:g}") / R_ring
    ax.plot(x, d, lw=1.3, color=BLUES[3], zorder=3)
    style(ax)
    ax.set_ylim(0, 3.0); ax.set_xlim(0, x[-1])
    ax.text(0.03, 0.93, f"peaks: {len(ring_peaks(d))}", transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
    ax.set_xticks([0, 256, 512])
    if sv != S_VALUES[0]:
        ax.tick_params(labelleft=False)
axes[1][0].set_ylabel("density  a/R", fontsize=10)
savefig(fig, "fig4_technology")

# ---------------- Fig 5: one landscape, two equilibria ----------------
mA, mB = state("siting_A"), state("siting_B")
fig = plt.figure(figsize=(7.0, 3.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.14)
axR = fig.add_subplot(gs[0])
axD = fig.add_subplot(gs[1], sharex=axR)
axR.fill_between(x, R_ring, color="#d9d9d9", lw=0, zorder=2)
for z in (450, 479):   # the two similar-capacity hills
    axR.annotate("", (z, R_ring[z]), xytext=(z, R_ring[z] + 0.9), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
axR.text(464, 2.7, "similar-capacity sites", fontsize=9, color=MUTED, ha="center")
axR.set_ylabel("capacity R", fontsize=9)
axR.tick_params(labelbottom=False)
style(axR)
axD.plot(x, mA / R_ring, lw=1.7, color=BLUES[3], zorder=3, label="start: the capacities")
axD.plot(x, mB / R_ring, lw=1.7, color="#b0561f", zorder=2, alpha=0.9, label="start: seeded near zone 479")
style(axD)
axD.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, handlelength=1.6)
axD.set_xlabel("zone")
axD.set_ylabel("density  a/R")
axD.set_xlim(0, x[-1])
savefig(fig, "fig5_siting")
