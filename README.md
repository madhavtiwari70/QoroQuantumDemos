# Divi Demo Console

Card #208: pick a demo, edit its data file, click Run, get real results
from real Divi execution. No notebook, no code changes.

## Repo structure — vendor vs. wrapper

```
streamlit_app.py           <- the UI
sync_demos.sh               <- refreshes vendor/ from upstream (run this, don't hand-edit vendor/)
data/
  <demo>.yaml                <- what someone actually edits to change a demo
demos/
  <demo>/<demo>_wrapper.py    <- our code: reads the data file, calls the vendor code
vendor/
  divi-demos/
    SYNC_INFO.md              <- when this was last synced, from which commit
    <demo>/                   <- exact copy of QoroQuantum/divi-demos, untouched
```

**The rule: never hand-edit anything under `vendor/`.** If upstream changes,
re-run `./sync_demos.sh` to refresh it. If you need different demo
parameters, edit `data/<demo>.yaml`. If you need different logic, edit the
matching `demos/<demo>/<demo>_wrapper.py` — never the vendor copy itself.

## Refreshing from upstream

```
./sync_demos.sh                    # sync all 6 demos
./sync_demos.sh spin_dynamics      # sync just one
```

This clones the real `QoroQuantum/divi-demos` repo fresh each time, copies
the relevant folders into `vendor/divi-demos/`, and records the exact
commit synced in `vendor/divi-demos/SYNC_INFO.md`. Review the diff with
`git status vendor/` before committing — if upstream renamed a function or
changed a signature, a wrapper might need a small update too.

## How to run it

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run streamlit run streamlit_app.py
```

## Status — all 10 demos individually verified with real Divi execution, against the vendor copy

Every scenario from all 6 original demo scripts is now covered — nothing left out.

| Demo | Category | Covers |
|---|---|---|
| Spin Dynamics (TFIM) | Time Evolution | Full demo |
| Economic Load Dispatch | Optimization · PCE-VQE | 3-generator scenario |
| Economic Load Dispatch — Six Generators | Optimization · PCE-VQE | 6-generator scenario (24 variables) |
| Quantum-Guided Cluster | QAOA | Full demo |
| Travelling Salesman | QAOA · QUBO | Part A: Direct QAOA |
| Travelling Salesman — Partitioned | QAOA · QUBO | Part B: Partitioned QAOA (larger instance) |
| Travelling Salesman — PCE Compression | QAOA · QUBO | Part C: PCE compression (qubit-compressed) |
| Minimum Birkhoff Decomposition | Optimization | Full demo |
| Portfolio Optimization | QAOA | Small synthetic portfolio (8 assets) |
| Portfolio Optimization — Full S&P 500 | QAOA | Real 2016 S&P 500 data (484 assets, partitioned) |

**Note on the Full S&P 500 demo:** this is a genuinely large job — 484 assets
partitioned into ~45 sub-problems, each solved independently. On local
simulation it took ~4 minutes even with iteration counts turned down for
testing; the default config in `data/portfolio_optimization_full.yaml` will
take noticeably longer. Consider warning whoever runs it live on a call
that this one takes a while, or turn down `qaoa.max_iterations` and
`qaoa.population_size` in that data file for a faster (less optimized) demo run.

## Known extra dependencies

`docplex` + `cplex` (Minimum Birkhoff Decomposition), `dimod` + `dwave-neal`
(Economic Load Dispatch, Portfolio Optimization) — all in `pyproject.toml`.
