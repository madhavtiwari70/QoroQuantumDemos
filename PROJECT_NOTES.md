# Divi Demo Console — Project Notes

Consolidated technical reference for the `QoroQuantumDemos` repo, written for
migration into a local agentic dev environment. Covers architecture,
decisions, gotchas, and open issues accumulated while building this.

**Verification status of this document:** every claim below was either
directly tested with real execution during development, or cross-checked
against the actual live files in the repo (`data/`, `demos/*/*_wrapper.py`,
`streamlit_app.py`, `sync_demos.sh`, `pyproject.toml`, `README.md`) as of
this writing.

---

## 1. What this is

A Streamlit app that lets someone pick one of 10 Divi quantum-computing
demos, edit its parameters in a plain-text config panel, click Run, and see
real results from a real Divi execution (local simulator or QoroService
cloud) — without touching Python code.

**Origin:** built to satisfy a task whose goal was: *"people take the
demo template, change a data file, and get a new demo without code changes
to Divi or the demo."* The original `QoroQuantum/divi-demos` repo was a set
of standalone scripts/notebooks with every parameter hardcoded — this
project is the config-driven wrapper layer built on top of it.

---

## 2. Architecture: vendor / wrapper / data

Every demo is split into three files, with a clean separation of what's
allowed to change and what isn't:

```
data/<demo>.yaml                     ← what a person actually edits
demos/<demo>/<demo>_wrapper.py        ← our code: reads the yaml, calls vendor code
vendor/divi-demos/<demo>/             ← untouched copy of QoroQuantum/divi-demos
```

**The rule:** `vendor/` is never hand-edited. It's refreshed only by running
`sync_demos.sh`, which clones the real upstream repo fresh and overwrites
the vendor copy wholesale, recording the exact commit synced in
`vendor/divi-demos/SYNC_INFO.md`.

**Why this shape, not a simpler one:**
- Keeps the actual Divi/quantum logic 100% traceable to Qoro's real,
  original code — nothing about the physics or algorithms was rewritten,
  only the hardcoded constants were extracted.
- If Qoro updates `divi-demos` upstream, re-running one script picks up the
  changes cleanly, without merge conflicts against our own customizations.
- Wrappers are small and single-purpose: unpack a dict, call the real
  function(s), shape the result for the UI.

**Wrapper pattern, concretely** (from `spin_dynamics_wrapper.py`):
```python
VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "spin_dynamics")
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

from spin_dynamics import build_tfim_hamiltonian, run_trajectory  # the REAL function
from plotting import plot_dynamics, plot_trajectory_error, absolute_trajectory_error

def run_from_config(cfg: dict, progress_callback=None) -> list[dict]:
    # cfg comes from yaml.safe_load() of the data file / edited text box
    ...
```

Every wrapper exposes exactly one function: `run_from_config(cfg, progress_callback=None) -> <result>`.
`streamlit_app.py` calls this uniformly for every demo; only the *rendering*
of the result differs per demo (see `DEMOS` registry below).

---

## 3. The `DEMOS` registry (in `streamlit_app.py`)

Single source of truth for what demos exist and how to run/display them:

```python
DEMOS = {
    "<Display Name>": {
        "category": "...",
        "data_file": "<demo>.yaml",         # in data/
        "folder": "<demo>",                  # in demos/
        "module": "<demo>_wrapper",          # imported dynamically
        "original_file": "<file>.py",        # shown in the source-viewer dialog
        "vendor_folder": "...",              # OPTIONAL: only if it differs from `folder`
        "no_single_source": True,            # OPTIONAL: for notebook-derived demos
    },
    ...
}
```

`vendor_folder` exists because several demos share one upstream source file
across multiple sidebar entries (e.g. Travelling Salesman's three scenarios
all point at the same `travelling_salesman.py`, but live in three different
`demos/<name>/` folders with three different wrapper functions and configs).

`no_single_source` exists for demos built from Jupyter notebook cells
(Portfolio Optimization, both variants) rather than one standalone script —
there's no single "original" file to show in the source viewer, so it shows
the wrapper itself instead, with an explanatory caption.

---

## 4. All 10 demos

| Display name | Category | Wrapper module | Covers | Scope note |
|---|---|---|---|---|
| Spin Dynamics (TFIM) | Time Evolution | `spin_dynamics_wrapper` | Full demo — all 3 physical regimes | — |
| Economic Load Dispatch | Optimization · PCE-VQE | `eld_wrapper` | 3-generator scenario | 6-gen is a separate entry |
| Economic Load Dispatch — Six Generators | Optimization · PCE-VQE | `eld_6gen_wrapper` | 6-generator (24 vars) | Uses simulated annealing (`classical_sa_solve`) for classical baseline instead of brute force — 24 vars is too many to brute-force |
| Quantum-Guided Cluster | QAOA | `qgc_wrapper` | Full demo | Cloud support lives *inside* the vendored `main.py`'s own `select_backend()`, not called directly in our wrapper |
| Travelling Salesman | QAOA · QUBO | `tsp_wrapper` | Part A: Direct QAOA | Small instance, 4 cities default |
| Travelling Salesman — Partitioned | QAOA · QUBO | `tsp_partitioned_wrapper` | Part B: Partitioned QAOA | Larger instance, uses `PartitioningProgramEnsemble` — see open issue #1 below |
| Travelling Salesman — PCE Compression | QAOA · QUBO | `tsp_pce_wrapper` | Part C: PCE compression | Same small instance as Part A, fewer qubits (e.g. 16→6 in testing) |
| Minimum Birkhoff Decomposition | Optimization | `birkhoff_wrapper` | Full demo | Uses a monkey-patch trick for cloud support — see §5 |
| Portfolio Optimization | QAOA | `portfolio_wrapper` | Small synthetic (8 assets) | Built from notebook cells, `no_single_source: True` |
| Portfolio Optimization — Full S&P 500 | QAOA | `portfolio_full_wrapper` | Real 2016 S&P 500 data (484 assets), partitioned | Slow (~4 min+ even scaled down); has an open cloud-execution bug — see §7 |

**Deliberate scope gaps, not bugs:** several original scripts run 2-3
scenarios in one file (e.g. the TSP script runs direct-QAOA, then
partitioned, then PCE, all in sequence). Rather than cram all three into
one confusing config schema, each scenario got its own sidebar entry +
data file + wrapper. All scenarios from all original scripts are now
covered — nothing was left out, just split cleanly.

---

## 5. Notable implementation techniques

### 5.1 The `sys.path` / `sys.modules` isolation fix

**The bug:** two different demos each ship a file named `plotting.py`
(`spin_dynamics/plotting.py` and `quantum_guided_cluster/plotting.py`) with
completely different contents. An earlier version of `streamlit_app.py`
added *every* demo folder to `sys.path` at startup — so `import plotting`
inside `spin_dynamics_original.py` could silently resolve to the wrong
file, causing `ImportError`/`ModuleNotFoundError` that looked unrelated to
the real cause.

**The fix**, now in `streamlit_app.py`:
```python
demos_root = os.path.join(HERE, "demos")
vendor_root = os.path.join(HERE, "vendor", "divi-demos")

# Remove any other demo folders from sys.path
sys.path[:] = [p for p in sys.path if not p.startswith(demos_root) and not p.startswith(vendor_root)]
# Drop any previously-imported module that came from a demo folder
for mod_name, mod in list(sys.modules.items()):
    mod_file = getattr(mod, "__file__", None) or ""
    if mod_file.startswith(demos_root) or mod_file.startswith(vendor_root):
        del sys.modules[mod_name]

sys.path.insert(0, os.path.join(demos_root, demo["folder"]))
module = importlib.import_module(demo["module"])
```
Only the *currently selected* demo's folder is ever visible to Python's
import system at once. **Lesson for future demos:** never add multiple
demo folders to `sys.path` simultaneously.

### 5.2 Monkey-patching a hardcoded backend (Birkhoff demo)

The vendored `minimum_birkhoff_decomposition/main.py` hardcodes
`MaestroSimulator(shots=5000)` inside its `main()` function — no backend
parameter exposed at all. To support switching to QoroService from the
data file *without hand-editing the vendored file*, the wrapper temporarily
redirects what `MaestroSimulator` resolves to inside that module's
namespace, right before calling it:
```python
import main as bmo
...
def backend_factory(shots=shots):
    if use_cloud:
        return QoroService(job_config=JobConfig(shots=shots))
    return MaestroSimulator(shots=shots)

original = bmo.MaestroSimulator
bmo.MaestroSimulator = backend_factory
try:
    bmo.main(args)
finally:
    bmo.MaestroSimulator = original
```
Python resolves names at call-time, so this safely redirects the call
without modifying the vendored source. Verified working for both local and
cloud paths (cloud path correctly raises the expected "no API key" error
when tested without credentials, confirming the redirect fires).

### 5.3 Argparse-driven original script → wrapper (Birkhoff)

`main.py` was already CLI-argument-driven (`argparse`), so the wrapper just
builds a `SimpleNamespace` mimicking the expected `args` object and calls
`main(args)` directly — no logic duplicated. Terminal output (including the
matrix breakdown) is captured via `contextlib.redirect_stdout` into a
string and returned for display via `st.code(...)`.

### 5.4 Grabbing a matplotlib figure after a plotting function that saves-to-file

Several vendored plotting functions (`plot_dynamics`, `plot_tour`,
`plot_comparison`, etc.) save a PNG to disk and don't return a `Figure`
object. Since each one calls `plt.figure()` internally before drawing,
`plt.gcf()` called immediately after the plotting function returns grabs
that exact figure, which can then be passed to `st.pyplot(fig)`:
```python
plot_dynamics(t_exact, m_exact, t_qdrift, m_qdrift, ..., filename="/tmp/....png")
fig_dynamics = plt.gcf()
```

### 5.5 `st.dialog` for the source viewer (not a sidebar expander)

Original implementation put the "view original code" panel in a sidebar
`st.expander` — cramped and unreadable at Streamlit's default sidebar
width, especially with line-numbered code. Replaced with a button that
opens a wide `st.dialog(..., width="large")` modal instead — readable when
open, takes zero permanent space when closed.

---

## 6. Environment / deployment gotchas (all encountered and fixed)

### 6.1 `.python-version` conflicting with `pyproject.toml`

**Symptom:** `error: The Python request from .python-version resolved to
Python 3.14.6, which is incompatible with the project's Python requirement:
>=3.11, <3.13` — `uv sync` aborts entirely, installing nothing (including
`pyyaml`), causing a downstream `ModuleNotFoundError: No module named 'yaml'`
that looks unrelated to the real cause.

**Cause:** the original Streamlit blank-app template ships a
`.python-version` file pinned to 3.14. `qoro-divi`'s dependency chain
(qiskit, dwave packages, numba, etc.) doesn't support 3.14 yet, so
`pyproject.toml` pins `requires-python = ">=3.11,<3.13"`. Two conflicting
version constraints in the same repo → `uv` refuses to proceed at all.

**Fix:** delete `.python-version` from the repo entirely. Let `uv` pick any
interpreter satisfying `pyproject.toml`'s range, rather than pinning one
exact patch version that may not exist on the deploy host.

### 6.2 Streamlit Cloud "more than one requirements file" — expected, not a bug

```
WARN: More than one requirements file detected in the repository. 
Available options: uv-sync .../uv.lock, poetry .../pyproject.toml. 
Used: uv-sync with .../uv.lock
```
This is just Streamlit Cloud telling you which dependency file it picked
(correctly, `uv.lock` — the more specific/locked one). Not an error.

### 6.3 Partial GitHub uploads causing `ModuleNotFoundError`

Repeatedly, files failed to make it into the repo via GitHub's web
drag-and-drop uploader — especially nested folder structures (30+ files
across `demos/<name>/` subfolders). Symptom: `ModuleNotFoundError: No
module named 'x_wrapper'` where the traceback shows Python couldn't even
*find* the file, not that it found it and failed inside it (no nested
stack frames below the import line).

**Diagnostic tip:** if a traceback for an import has zero nested frames
below it, the file is very likely missing from the repo, not broken.

**Reliable fix:** package everything as a single `.zip`, unzip locally, and
push the whole tree via `git add . && git commit && git push` (or drag the
already-correctly-nested unzipped folder into GitHub's uploader in one
shot) — never upload large nested trees file-by-file via the web UI.

### 6.4 Notebook execution: missing `execution_count` field

When executing a vendored `.ipynb` headlessly via `nbconvert`, some code
cells were missing the `execution_count` field that `nbformat`'s validator
requires, causing a validation failure before execution even started.
Fixed by normalizing the notebook first:
```python
import nbformat
nb = nbformat.read(entry, as_version=4)
for cell in nb.cells:
    if cell.get("cell_type") == "code" and "execution_count" not in cell:
        cell["execution_count"] = None
nbformat.write(nb, entry)
```

### 6.5 Cross-demo Python module name collisions

Covered in §5.1 — listed again here because it's an *environment*-shaped
gotcha (only reproduces when multiple demos coexist in one running
process), not something visible testing one wrapper in isolation.

### 6.6 Upstream repo changed mid-project

On **Aug 3, 2026**, `QoroQuantum/divi-demos` was refactored for "Divi
0.13" (up from 0.12.1) — `cluster_maxcut` and `molecular_ground_state`
demos were **removed** from the repo (the repo's own README was not
updated to match, and still lists them as of this writing). This project's
demo set reflects the 6 demos that exist in the *current* upstream repo,
not the historical 8. Any future `sync_demos.sh` run should be checked
against `SYNC_INFO.md`'s recorded commit to catch further upstream drift.

---

## 7. Full S&P 500 Demo: Root Cause Diagnosis & Resolution

**Status: RESOLVED.**

The "Portfolio Optimization — Full S&P 500" demo suffered from two distinct failure modes:

### 7.1 Local Execution: Silent UI Freeze (~18 minutes)
- **Root Cause:** The default parameters in `data/portfolio_optimization_full.yaml` were over-dimensioned for interactive demo use (`max_iterations: 10`, `population_size: 30`, `shots: 10000`, 45 partitions = 13,500 circuit simulations). A single iteration required ~107 seconds, leading to an 18-minute total runtime. Because `portfolio_full_wrapper.py` only emitted a single static progress callback before blocking on `ensemble.run().join()`, Streamlit appeared completely frozen and suffered websocket timeouts.
- **Resolution:**
  1. Default parameters were tuned for interactive demo responsiveness in `data/portfolio_optimization_full.yaml` (`max_iterations: 2`, `population_size: 10`, `shots: 1000`, `early_stopping_patience: 2`), reducing local runtime to ~45 seconds while allowing users to easily scale up in the UI editor.
  2. Non-blocking execution `ensemble.run(blocking=False)` with a real-time polling loop over `ensemble.futures` streams granular progress updates (`Optimizing partitions: X/45 completed (Y%)...`) directly to Streamlit via `progress_callback`.

### 7.2 Cloud Execution: `KeyError: 0` in `_grouping.py`
- **Root Cause:** In `divi/qprog/ensemble.py`, `PartitioningProgramEnsemble.run()` defaulted to `BatchConfig(mode=BatchMode.MERGED)`. When merging 45 sub-problems with distinct Hamiltonians, `QoroService` chunked the large batch (>0.95MB) across multiple HTTP submissions. Unlike shot groups, the chunker did not recalculate `circuit_ham_map` slice offsets, and pagination over 1,350+ circuits dropped results. When operator `pos=0` was omitted from returned results, post-processing skipped indexing, leaving Pauli string keys; `_grouping.py:106` then crashed with `KeyError: 0` attempting `val[0]`.
- **Resolution:**
  - Configured `BatchConfig(mode=BatchMode.OFF)` for cloud execution (or when `batch_mode: "off"` is requested). This isolates each partition into its own independent job submission with a single clean Hamiltonian, bypassing the multi-Hamiltonian chunking bug and large pagination dropouts entirely.

### 7.3 Financial Metrics & Defensive Enhancements
- Wrapped `ensemble.join()` with robust exception handling and actionable troubleshooting tips.
- Added `_compute_portfolio_metrics` from `utils.py` to calculate Return, Risk (Variance), and Sharpe Ratio, rendering them in `streamlit_app.py` alongside selected asset indices.
- Added `n_best_sets = min(qaoa_cfg["n_best_sets"], pop_size)` to prevent `ValueError` if a user configures small population sizes.
- Configured `MPLCONFIGDIR` to suppress non-writable cache warnings.

---

## 8. Deployment reference

### Running locally
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run streamlit run streamlit_app.py
```

### QoroService cloud credentials
- Sign up at `dash.qoroquantum.net`, generate a token on the Token page
- Local: create `.env` at repo root with `QORO_API_KEY="..."` (already
  covered by `.gitignore` — never commit this file)
- Streamlit Community Cloud: set the same key in **Manage app → Settings →
  Secrets** as `QORO_API_KEY = "..."`
- No other config needed — `QoroService`'s endpoint is hardcoded inside the
  `qoro-divi` library itself; the token alone identifies the account

### Streamlit Community Cloud specifics
- Free tier has **no custom domain** support (e.g. `demo.qoroquantum.com`
  is not possible without a paid tier or self-hosting elsewhere). A free
  **custom subdomain** (`your-name.streamlit.app` instead of a random
  hash) *is* available under Settings → General.
- Set the Python version explicitly in the app's own Streamlit Cloud
  settings as a second line of defense, in addition to `pyproject.toml`'s
  `requires-python`.
- A plain `git push` does **not** always force Streamlit to re-import
  already-loaded Python modules within the same running process — if a
  code change doesn't seem to take effect, use **Reboot app** from the
  "⋮" menu, not just wait for auto-refresh.

### Known extra dependencies (already in `pyproject.toml`)
- `docplex` + `cplex` — required only by Minimum Birkhoff Decomposition
- `dimod` + `dwave-neal` — required by Economic Load Dispatch (both
  variants) and Portfolio Optimization (both variants)
- `qoro-divi[jupyter]==0.13.0` — pinned to match the version the vendored
  `divi-demos` scripts were written against (Aug 2026 "Divi 0.13" refactor)

---

## 9. Extending this project

**To add a new demo, or a new scenario of an existing one:**
1. Identify the relevant function(s) in the vendored original script
   (`vendor/divi-demos/<name>/...`) — never write new quantum logic from
   scratch if the vendored code already has it
2. Create `data/<new-name>.yaml` with every tunable parameter
3. Create `demos/<new-name>/<new-name>_wrapper.py` with one function,
   `run_from_config(cfg, progress_callback=None)`, that unpacks `cfg` and
   calls the real vendored function(s) — follow the `_resolve_backend(cfg)`
   pattern used everywhere else for local/cloud switching
4. Add one entry to `DEMOS` in `streamlit_app.py`
5. Add one `elif selected_label == "..."` render branch for the result
   shape your wrapper returns
6. Smoke-test standalone with a shrunk config before wiring into the UI —
   every wrapper in this project was verified this way before being added

**To resync from upstream:** `./sync_demos.sh` (or `./sync_demos.sh
<one-demo-name>`), then `git status vendor/` to review the diff before
committing — if Qoro renamed a function or changed a signature upstream, a
wrapper may need a small matching update.
