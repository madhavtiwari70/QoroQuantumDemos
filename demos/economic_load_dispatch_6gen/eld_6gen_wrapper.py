"""
Economic Load Dispatch — Six Generators — config-driven wrapper.

Covers the "Six-generator variant" section from
vendor/divi-demos/economic_load_dispatch/economic_load_dispatch.py.
Reuses the actual functions unmodified: build_qubo, classical_sa_solve,
solve_with_pce, find_best_repaired_solution. Uses simulated annealing
(not brute force) for the classical baseline, since 24 variables is too
many to brute-force exhaustively — same choice the original script makes.
"""

import sys
import os

VENDOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "economic_load_dispatch"
)
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

from economic_load_dispatch import (
    build_qubo,
    classical_sa_solve,
    solve_with_pce,
    find_best_repaired_solution,
)
from divi.backends import MaestroSimulator, QoroService, JobConfig


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))
    return MaestroSimulator(shots=b["shots"])


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    import time

    generators = cfg["generators"]
    demand = cfg["demand"]
    pce_cfg = cfg["pce"]
    backend = _resolve_backend(cfg)

    if progress_callback:
        progress_callback(f"Building QUBO ({len(generators)} generators)...")
    bqm, var_names = build_qubo(generators, demand=demand)

    if progress_callback:
        progress_callback("Computing classical baseline (simulated annealing)...")
    classical_result = classical_sa_solve(
        generators, demand, bqm, num_reads=cfg["classical"]["num_reads"]
    )

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local MaestroSimulator"
        progress_callback(f"Running PCE-VQE on {backend_name}...")

    t0 = time.time()
    pce_solver = solve_with_pce(
        bqm,
        n_layers=pce_cfg["n_layers"],
        max_iterations=pce_cfg["max_iterations"],
        population_size=pce_cfg["population_size"],
        backend=backend,
    )
    runtime = time.time() - t0

    result = find_best_repaired_solution(pce_solver, bqm, generators, demand)

    output = {
        "n_generators": len(generators),
        "n_variables": len(var_names),
        "n_qubits": pce_solver.n_qubits,
        "classical_cost": classical_result[-1] if classical_result else None,
        "classical_powers": classical_result[:-1] if classical_result else None,
        "runtime_s": runtime,
        "quantum_result": None,
    }

    if result is not None:
        powers, cost, prob = result
        output["quantum_result"] = {
            "powers": powers,
            "cost": cost,
            "seed_probability": prob,
        }

    return output
