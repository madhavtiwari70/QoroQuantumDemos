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

## Status — all individually verified with real Divi execution, against the vendor copy

| Demo | Category | Covers |
|---|---|---|
| Spin Dynamics (TFIM) | Time Evolution | Full demo |
| Economic Load Dispatch | Optimization · PCE-VQE | 3-generator scenario |
| Quantum-Guided Cluster | QAOA | Full demo |
| Travelling Salesman | QAOA · QUBO | "Part A: Direct QAOA" only |
| Minimum Birkhoff Decomposition | Optimization | Full demo |
| Portfolio Optimization | QAOA | Small synthetic portfolio only |

## Known extra dependencies

`docplex` + `cplex` (Minimum Birkhoff Decomposition), `dimod` + `dwave-neal`
(Economic Load Dispatch, Portfolio Optimization) — all in `pyproject.toml`.
