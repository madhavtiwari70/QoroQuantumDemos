"""
Portfolio Optimization — Full S&P 500 — config-driven wrapper.

Covers the real-data, partitioned-QAOA part of vendor/divi-demos/
portfolio_optimization/portfolio_optimization.ipynb. Reuses
utils.py's build_full_portfolio_qubo unmodified, and loads the real
S&P 500 data files (2016-01-01_*.npy) shipped in the same vendor folder.
"""

import sys
import os
import io
import contextlib

# Suppress matplotlib font cache rebuild / permission warnings
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "matplotlib"),
)

VENDOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "portfolio_optimization"
)
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

import numpy as np

from utils import build_full_portfolio_qubo, _compute_portfolio_metrics
from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.qprog import BeamSearchStrategy, EarlyStopping
from divi.qprog.ensemble import BatchConfig, BatchMode
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer
from divi.qprog.workflows import PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
import hybrid


def _resolve_backend(cfg: dict):
    b = cfg.get("backend", {})
    shots = b.get("shots", 1000)
    if b.get("use_cloud", False):
        return QoroService(job_config=JobConfig(shots=shots))
    return MaestroSimulator(shots=shots)


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    import time

    if progress_callback:
        progress_callback("Loading real S&P 500 data (2016-01-01)...")

    real_returns = np.load(os.path.join(VENDOR_DIR, "2016-01-01_returns.npy"))
    real_covariance = np.load(os.path.join(VENDOR_DIR, "2016-01-01_covariance.npy"))

    n_assets = len(real_returns)

    if progress_callback:
        progress_callback(f"Building QUBO for {n_assets} assets...")

    real_qubo = build_full_portfolio_qubo(real_returns, real_covariance, lambda_param=cfg["lambda_param"])

    part_cfg = cfg["partitioning"]
    qaoa_cfg = cfg["qaoa"]
    agg_cfg = cfg["aggregation"]
    backend = _resolve_backend(cfg)

    problem = BinaryOptimizationProblem(
        real_qubo,
        decomposer=CommunityDecomposer(
            max_cluster_size=part_cfg["max_partition_size"],
            method=part_cfg["method"],
            seed=part_cfg["seed"],
        ),
        composer=hybrid.SplatComposer(),
    )

    pop_size = qaoa_cfg["population_size"]
    n_best_sets = min(qaoa_cfg.get("n_best_sets", 3), pop_size)

    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=qaoa_cfg["n_layers"],
        optimizer=MonteCarloOptimizer(
            population_size=pop_size,
            n_best_sets=n_best_sets,
        ),
        max_iterations=qaoa_cfg["max_iterations"],
        early_stopping=EarlyStopping(patience=qaoa_cfg["early_stopping_patience"]),
        backend=backend,
    )

    if progress_callback:
        progress_callback("Decomposing portfolio into partitions...")
    ensemble.create_programs()
    n_partitions = len(ensemble.programs)

    use_cloud = cfg.get("backend", {}).get("use_cloud", False)
    backend_name = "QoroService" if use_cloud else "local MaestroSimulator"

    if progress_callback:
        progress_callback(
            f"Running {n_partitions} partitions in parallel on {backend_name}..."
        )

    # Cloud execution requires BatchMode.OFF because merging 45 Hamiltonians
    # into a single request triggers large-payload chunking and results pagination
    # misalignments in QoroService. BatchMode.OFF isolates each partition.
    batch_mode_str = str(cfg.get("batch_mode", "")).lower()
    if batch_mode_str == "off":
        batch_config = BatchConfig(mode=BatchMode.OFF)
    elif batch_mode_str == "merged":
        batch_config = BatchConfig(mode=BatchMode.MERGED)
    else:
        batch_config = BatchConfig(mode=BatchMode.OFF if use_cloud else BatchMode.MERGED)

    t0 = time.time()
    try:
        ensemble.run(blocking=False, batch_config=batch_config)

        # Stream real-time partition progress
        while not all(f.done() for f in ensemble.futures):
            done_count = sum(1 for f in ensemble.futures if f.done())
            pct = int((done_count / n_partitions) * 100)
            if progress_callback:
                progress_callback(
                    f"Optimizing partitions: {done_count}/{n_partitions} completed ({pct}%)..."
                )
            time.sleep(0.4)

        if progress_callback:
            progress_callback(
                f"Optimizing partitions: {n_partitions}/{n_partitions} completed (100%). Collecting results..."
            )

        ensemble.join()
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error during execution on {backend_name}: {e}")
        raise RuntimeError(
            f"Partitioning QAOA ensemble failed on {backend_name}: {e}\n"
            f"Tip: If running on cloud, ensure BatchMode is OFF and your Qoro API key has sufficient quota."
        ) from e

    runtime = time.time() - t0

    if progress_callback:
        progress_callback("Aggregating partition results via beam search...")

    solution, energy = ensemble.aggregate_results(
        BeamSearchStrategy(
            beam_width=agg_cfg["beam_width"],
            n_partition_candidates=agg_cfg["n_partition_candidates"],
        )
    )

    solution_arr = np.array(solution, dtype=int)
    n_selected = int(solution_arr.sum())
    selected_assets = np.where(solution_arr == 1)[0].tolist()

    portfolio_return, portfolio_risk, sharpe_ratio = _compute_portfolio_metrics(
        real_returns, real_covariance, solution_arr
    )

    return {
        "n_assets": n_assets,
        "n_partitions": n_partitions,
        "total_circuit_count": ensemble.total_circuit_count,
        "n_selected": n_selected,
        "selected_assets": selected_assets,
        "energy": float(energy),
        "portfolio_return": float(portfolio_return),
        "portfolio_risk": float(portfolio_risk),
        "sharpe_ratio": float(sharpe_ratio),
        "runtime_s": runtime,
    }
