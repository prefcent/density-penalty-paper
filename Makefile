# Reproduce the preprint's figures.
#   make runs          solve every run and rewrite archive/ (states, manifests, summaries)
#   make sweeps        solve the multi-start sweeps and rewrite archive/sweeps/ (resumable, work/)
#   make figs          render figs/fig1..fig8 from archive/
#   make check         re-run Figures 1-5 and a sample of sweep runs; compare with archive/
#   make check-sweeps  re-solve every sweep run and compare with archive/sweeps/
.PHONY: runs sweeps figs check check-sweeps
runs:
	python3 run_fig1.py && python3 run_fig2.py && python3 run_fig3.py && python3 run_fig4.py && python3 run_fig5.py
sweeps:
	python3 sweeps.py run
figs:
	python3 make_figs.py && python3 make_sweep_figs.py
check:
	python3 check_archive.py
check-sweeps:
	python3 check_archive.py --sweeps
