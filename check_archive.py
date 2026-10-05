"""Re-run archived runs and compare them with the archive.

Figures 1-5: every archived run is repeated with the same inputs and settings (the run
scripts' protocol, including continuation order) and its result hash is compared with
the archived one.

Sweeps (Figures 6-8, Table 2): by default a fixed sample -- every 50th run of each sweep
in name order -- is re-solved and its result hash compared with the archived record;
``--sweeps`` re-solves every sweep run (about 20,000 runs). Budget-exhausted runs compare
their returned state too; domain-stopped runs must stop again.

A mismatch is reported per run; the archive itself is never modified.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import common

ARCHIVE = Path("archive")


def check_sweeps(full: bool) -> int:
    import gzip
    from concurrent.futures import ProcessPoolExecutor

    # Workers re-import sweeps when processes are spawned rather than forked (the
    # default on macOS, and on Linux from Python 3.14), so the scratch directory
    # goes through the environment, which they inherit, not a module attribute.
    work = tempfile.mkdtemp(prefix="preprint-check-sweeps-")
    os.environ["SWEEP_WORK"] = work
    import sweeps as sw
    sw.WORK = Path(work)
    failures = 0
    for sweep in sw.SWEEPS:
        with gzip.open(sw.ARCHIVE / f"{sweep}.runs.jsonl.gz", "rt") as fh:
            recs = [json.loads(line) for line in fh]
        sample = recs if full else recs[::50]
        (sw.WORK / sweep).mkdir(parents=True)
        args = [(sweep, r["land"], r["gamma"], r["s"], r["kappa"], r["start"]) for r in sample]
        with ProcessPoolExecutor(max_workers=sw.default_workers()) as ex:
            fresh = list(ex.map(sw.solve_one, *zip(*args), chunksize=4))
        bad = []
        for old, new in zip(sample, fresh):
            if old["status"] != new["status"]:
                bad.append(old["run"])
            elif old["status"] != "DOMAIN_STOP" and (old["manifest"]["observations"]["result_sha256"]
                                                     != new["manifest"]["observations"]["result_sha256"]):
                bad.append(old["run"])
        failures += len(bad)
        print(f"  {sweep:12s} {len(sample):6d} runs re-solved: "
              f"{'identical' if not bad else f'{len(bad)} DIFFERENT, e.g. {bad[:3]}'}", flush=True)
    shutil.rmtree(sw.WORK)
    return failures


def main() -> int:
    import sys
    sweeps_only = "--sweeps" in sys.argv
    if sweeps_only:
        failures = check_sweeps(full=True)
        print("all sweep runs reproduced" if failures == 0 else f"{failures} sweep run(s) differ")
        return 1 if failures else 0
    tmp = Path(tempfile.mkdtemp(prefix="preprint-check-"))
    common.ARCHIVE = str(tmp)            # the run scripts write through this name
    import run_fig1, run_fig2, run_fig3, run_fig4, run_fig5   # noqa: E402
    for mod in (run_fig1, run_fig2, run_fig3, run_fig4, run_fig5):
        mod.ARCHIVE = str(tmp)
        mod.main()
    failures = 0
    names = sorted(p.name[: -len(".manifest.json")] for p in ARCHIVE.glob("*.manifest.json"))
    print(f"\ncomparing {len(names)} runs:", flush=True)
    for name in names:
        with open(ARCHIVE / f"{name}.manifest.json") as fh:
            archived = json.load(fh)
        with open(tmp / f"{name}.manifest.json") as fh:
            fresh = json.load(fh)
        same_hash = archived["observations"]["result_sha256"] == fresh["observations"]["result_sha256"]
        diff = float(np.max(np.abs(np.load(ARCHIVE / f"{name}.mass.npy") - np.load(tmp / f"{name}.mass.npy"))))
        ok = same_hash and diff == 0.0
        failures += not ok
        print(f"  {name:32s} {'identical' if ok else 'DIFFERENT'}  (max |diff| {diff:.1e}; "
              f"archived {archived['fingerprint']['package_version']}, "
              f"now {fresh['fingerprint']['package_version']})", flush=True)
    shutil.rmtree(tmp)
    print("\nsweeps (sample):", flush=True)
    failures += check_sweeps(full=False)
    print("all archived states reproduced" if failures == 0 else f"{failures} run(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
