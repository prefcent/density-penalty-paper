# density-penalty-paper: reproduction archive

Repository `prefcent/density-penalty-paper`. Scripts, archived states and manifests behind the eight figures of the note
*Preferential centrality with a density penalty: formulation and an open-source
implementation* (Hellervik, 2026). The manuscript itself is not part of this
archive; the solver is the [`prefcent`](https://github.com/prefcent/prefcent) package.

Every run in `archive/` was produced by the scripts here, and re-running the scripts
with the pinned solver release reproduces every archived state bit for bit.
`make check` re-solves the 23 figure runs and a fixed sample of the sweep runs (every
50th); `make check-sweeps` re-solves all 19,692 sweep runs.

## Solver release

Every run in `archive/` was solved with **prefcent 0.1.1**
([PyPI](https://pypi.org/project/prefcent/0.1.1/),
[GitHub release](https://github.com/prefcent/prefcent/releases/tag/v0.1.1), version DOI
[10.5281/zenodo.23103539](https://doi.org/10.5281/zenodo.23103539)), installed from the
same wheel that is on PyPI (SHA-256 `70f36bda…`, recorded in `release.json`). The 23
figure runs were regenerated on 2026-09-17 with 16 BLAS threads; the multi-start sweeps
(`archive/sweeps/`, 19,692 runs) were solved on 2026-10-02 with one BLAS thread per
worker. Each manifest records version 0.1.1 and the verified installed artifact identity
(the hash of the installed `RECORD`), the BLAS backend and the thread count.

`release.json` identifies the exact wheel and source distribution, the source commits,
the numerical environment, the archive files, and the checks run before release. The
solver version cited by both the preprint and browser companion is **0.1.1**.

## Layout

| path | contents |
|---|---|
| `common.py` | the two landscapes, the two kernels, the fixed peak/center rule, and shared run helpers |
| `torus_fft.py` | the torus kernel applied by FFT, passed to prefcent as an `OperatorKernel` (used by the sweeps) |
| `sweeps.py` | the multi-start sweeps: landscapes, initial states, grids, solving, counting, perturbation-return tests, packing |
| `make_sweep_figs.py` | renders `figs/fig6..fig8` and `figs/table2_sweeps.txt` from `archive/sweeps/` (no solving) |
| `run_fig1.py` … `run_fig5.py` | one script per figure; each solves its runs and writes states, manifests and a summary to `archive/` |
| `make_figs.py` | renders `figs/fig1..fig5.{pdf,png}` from `archive/` (no solving) |
| `check_archive.py` | re-runs everything into a temporary directory and compares result hashes with `archive/` |
| `archive/sweeps/` | per sweep: `<sweep>.runs.jsonl.gz` (every run's record and manifest), `<sweep>.states.npz` (every distinct state found), `<sweep>.summary.json` (grid, protocol, per-point counts, perturbation-return results) |
| `archive/` | `<run>.mass.npy` (the returned state), `<run>.manifest.json` (the run's provenance manifest), `fig<N>_summary.json` (the numbers quoted in the captions), `fig3_hills_R.npy` (the four-hill capacity landscape) |
| `figs/` | the rendered figures |

## Runs per figure

| figure | runs | landscape | settings |
|---|---|---|---|
| 1 | `fig1_ring_g2_k{0,0.0128}_{b200,b1000,tol}` | ring | γ = 2, ω = 1, start a = R; budgets 200 and 1,000 (no tolerance), and to tolerance 1e-12 |
| 2 | `ring512_kappa{0,0.0051,0.0128,0.0255}` | ring | γ = 0.5; κ = 0 at ω = 1, then continuation in κ at ω = 0.5 |
| 3 | `fig3_hills_g1.5_k{0.1,0.05,0.02}` | four-hill torus | γ = 1.5; continuation downward in κ at ω = 0.5 (0.05 is the chain's middle step) |
| 4 | `decay_torus_s{10,3,1,0.3}`, `decay_ring_s{10,3,1,0.3}` | both | line-haul costs scaled by s, terminal penalty fixed; continuation in s at ω = 0.5 |
| 5 | `siting_A`, `siting_B` | ring | γ = 1.75, κ = 0.0128, ω = 0.5; two starts on identical inputs, plus perturbation-return checks |

### Multi-start sweeps (Figures 6-8, Table 2)

| sweep | landscapes | gamma | s | kappa | starts | runs |
|---|---|---|---|---|---|---|
| `prelim_k0.05`, `prelim_k0.02`, `prelim_k0.01` | 3 random at sigma = 0.5, four-hill | 0.5-2 (7) | 0.3-10 (7) | one each | 8 | 1,568 each |
| `amplitude` | 3 random, sigma 0.01-0.2 (5) | 0.5-2 (7) | 1 | 0.02 | 12 | 1,260 |
| `probe` | 3 random, sigma 0.01, 0.05 | 2.5, 3, 4 | 1, 3 | 0.01, 0.005 | 4 | 288 |
| `grid_sigma` (Figure 7) | 3 random, sigma 0.01-0.5 (7) | 0.5-4 (10) | 1 | 0.01, 0.005 | 16 | 6,720 |
| `grid_s` (Figures 6, 8) | 3 random, sigma 0.1, 0.5 | 0.5-4 (10) | 0.3-10 (7) | 0.01 | 16 | 6,720 |

The protocol (initial states, grouping tolerance 1e-6 in the normalized max norm,
perturbation-return test) is stated in `sweeps.py` and in the note's appendix. The
sweeps use the FFT torus kernel and one BLAS thread per worker; each manifest records
both. `make sweeps` solves into `work/` (resumable, not archived) and packs
`archive/sweeps/`; it takes about 30 minutes on 60 cores.

Common to all runs: decay `(c + d0)^(-2)` with `d0 = 5000` metre-equivalents, no
self-interaction, `ρ = 2`, zones 1 km apart, mean capacity 1, tolerance 1e-12 on the
normalized damped step where a tolerance is used. The peak/center rule counts maxima
of the density `a/R` above 1.3: strict local maxima on the ring, and 9×9-neighborhood
maxima on the torus, the neighborhoods wrapping around the torus edges.

## Usage

```bash
pip install -r requirements.txt
make figs     # render from the archive (seconds)
make check    # re-solve Figures 1-5 and a sample of sweep runs, compare with the archive
make check-sweeps  # re-solve every sweep run (about 20,000) and compare
make runs     # re-solve Figures 1-5 and overwrite their archive
make sweeps   # re-solve the sweeps and overwrite archive/sweeps/
```

The native archive uses Python 3.12, NumPy 2.5.2, SciPy 1.18.1, and
OpenBLAS with 16 threads. Set `OPENBLAS_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, and
`MKL_NUM_THREADS=16` before running Python, matching the manifest environment. The
requirements pin the native scientific stack; the browser companion locks its
WebAssembly stack separately and compares numerically rather than by native result
hash. Bit-for-bit repeatability also depends on the recorded platform and backend.

## Use of AI tools

AI tools were used extensively in producing this archive and the note it supports.
Claude models (Anthropic; Claude Opus 4.8, Opus 5, Opus 5.5 and Fable 5), working as
coding agents through Claude Code, wrote the run, sweep, check and rendering scripts in
this repository, ran every computation in `archive/`, and rendered the figures. They also
wrote most of the `prefcent` solver to the author's specification, drafted and revised
the note's text, and checked its citations against local copies of the sources. Cursor's
coding agent implemented parts of the solver's first milestones, and OpenAI Codex
implemented fixes found in a release review. OpenAI GPT-6 models reviewed drafts of the
note and of the software; the author decided which findings to adopt.

The author formulated the model and the density penalty, made the design and scientific
decisions, and reviewed all of the `prefcent` source code and all of the scripts in
this repository. The author has reviewed, and takes
responsibility for, all content of the note. The archive is built so that none of this
has to be taken on trust: `make check` and `make check-sweeps` re-solve the runs and
compare their result hashes with the archived manifests.

## License

MIT (see `LICENSE`).
