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

VENDOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "portfolio_optimization"
)
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

import numpy as np

from utils import build_full_portfolio_qubo
from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.qprog import BeamSearchStrategy, EarlyStopping
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer
from divi.qprog.workflows import PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
import hybrid


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))
    return MaestroSimulator(shots=b["shots"])


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

    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=qaoa_cfg["n_layers"],
        optimizer=MonteCarloOptimizer(
            population_size=qaoa_cfg["population_size"],
            n_best_sets=qaoa_cfg["n_best_sets"],
        ),
        max_iterations=qaoa_cfg["max_iterations"],
        early_stopping=EarlyStopping(patience=qaoa_cfg["early_stopping_patience"]),
        backend=backend,
    )

    if progress_callback:
        progress_callback("Decomposing portfolio into partitions...")
    ensemble.create_programs()
    n_partitions = len(ensemble.programs)

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local MaestroSimulator"
        progress_callback(
            f"Running {n_partitions} partitions in parallel on {backend_name}..."
        )

    t0 = time.time()
    ensemble.run().join()
    runtime = time.time() - t0

    solution, energy = ensemble.aggregate_results(
        BeamSearchStrategy(
            beam_width=agg_cfg["beam_width"],
            n_partition_candidates=agg_cfg["n_partition_candidates"],
        )
    )

    n_selected = int(solution.sum())

    return {
        "n_assets": n_assets,
        "n_partitions": n_partitions,
        "total_circuit_count": ensemble.total_circuit_count,
        "n_selected": n_selected,
        "energy": energy,
        "runtime_s": runtime,
    }
