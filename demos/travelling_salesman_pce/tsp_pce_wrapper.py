"""
Travelling Salesman — PCE Compression (Part C) — config-driven wrapper.

Covers "Part C: PCE compression" from
vendor/divi-demos/travelling_salesman/travelling_salesman.py. Reuses the
actual functions unmodified: generate_cities, compute_distance_matrix,
build_tsp_qubo, classical_brute_force, solve_with_pce.
"""

import sys
import os

VENDOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "travelling_salesman"
)
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

from travelling_salesman import (
    generate_cities,
    compute_distance_matrix,
    build_tsp_qubo,
    classical_brute_force,
    solve_with_pce,
)
from divi.backends import MaestroSimulator, QoroService, JobConfig


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))
    return MaestroSimulator(shots=b["shots"])


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    import time

    n_cities = cfg["n_cities"]
    seed = cfg["seed"]
    pce_cfg = cfg["pce"]
    backend = _resolve_backend(cfg)

    if progress_callback:
        progress_callback(f"Generating {n_cities}-city instance...")
    cities = generate_cities(n_cities, seed=seed)
    dist_matrix = compute_distance_matrix(cities)
    bqm, _ = build_tsp_qubo(dist_matrix)

    if progress_callback:
        progress_callback("Computing classical optimum...")
    classical_tour, classical_dist = classical_brute_force(dist_matrix)

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local MaestroSimulator"
        progress_callback(f"Running PCE-compressed solve on {backend_name}...")

    t0 = time.time()
    pce_tour, pce_dist, pce_qubits = solve_with_pce(
        bqm,
        dist_matrix,
        n_layers=pce_cfg["n_layers"],
        max_iterations=pce_cfg["max_iterations"],
        alpha=pce_cfg["alpha"],
        population_size=pce_cfg["population_size"],
        shots=cfg["backend"]["shots"],
        encoding_type=pce_cfg["encoding_type"],
        backend=backend,
    )
    runtime = time.time() - t0

    return {
        "n_cities": n_cities,
        "direct_qubits": n_cities ** 2,
        "pce_qubits": pce_qubits,
        "classical_tour": classical_tour,
        "classical_distance": classical_dist,
        "pce_tour": pce_tour,
        "pce_distance": pce_dist,
        "ratio": pce_dist / classical_dist if classical_dist else None,
        "runtime_s": runtime,
    }
