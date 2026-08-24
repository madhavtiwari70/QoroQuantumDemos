"""
Travelling Salesman — Partitioned (Part B) — config-driven wrapper.

Covers "Part B: Partitioned QAOA on a larger instance" from
vendor/divi-demos/travelling_salesman/travelling_salesman.py. Reuses the
actual functions unmodified: generate_cities, compute_distance_matrix,
build_tsp_qubo, classical_brute_force, solve_partitioned_tsp, plot_comparison.

This is a separate demo from "Travelling Salesman" (Part A) — see that
demo's wrapper for the small direct-QAOA scenario.
"""

import sys
import os
import matplotlib.pyplot as plt

VENDOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "travelling_salesman"
)
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

from travelling_salesman import (
    generate_cities,
    compute_distance_matrix,
    build_tsp_qubo,
    classical_brute_force,
    solve_partitioned_tsp,
    plot_comparison,
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
    qaoa_cfg = cfg["qaoa"]
    part_cfg = cfg["partitioning"]
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
        progress_callback(f"Running partitioned QAOA on {backend_name}...")

    t0 = time.time()
    quantum_tour, quantum_dist = solve_partitioned_tsp(
        bqm,
        dist_matrix,
        decomposer_size=part_cfg["decomposer_size"],
        n_layers=qaoa_cfg["n_layers"],
        max_iterations=qaoa_cfg["max_iterations"],
        shots=cfg["backend"]["shots"],
        backend=backend,
    )
    runtime = time.time() - t0

    plot_comparison(cities, classical_tour, classical_dist, quantum_tour, quantum_dist)
    fig = plt.gcf()

    return {
        "classical_tour": classical_tour,
        "classical_distance": classical_dist,
        "quantum_tour": quantum_tour,
        "quantum_distance": quantum_dist,
        "ratio": quantum_dist / classical_dist if classical_dist else None,
        "runtime_s": runtime,
        "figure": fig,
    }
