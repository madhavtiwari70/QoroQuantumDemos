"""
Quantum-Guided Cluster Algorithm for Max-Cut — benchmark runner.

Generates a graph, runs SA / coupling-constant / quantum-guided cluster
algorithms (QAOA at any depths plus optional PCE encodings), and saves the
comparison plots.

Reference: arXiv:2508.10656 (Eder et al., AWS Quantum Solutions Lab).
"""

import os
import time

import numpy as np

from algorithm import (
    ClusterAlgoResult,
    generate_random_maxcut_graph,
    ising_energy,
    extract_qaoa_correlations,
    extract_pce_correlations,
    coupling_constant_correlations,
    correlation_guided_cluster_algorithm,
    simulated_annealing,
)
from plotting import (
    plot_approximation_ratios,
    plot_correlation_heatmaps,
    plot_circuit_efficiency,
    plot_energy_distribution,
)

from divi.backends import MaestroSimulator


def summarize_approximation_ratios(ratios_by_seed: dict[int, list[float]]) -> dict:
    """Return mean and range summaries for repeated benchmark ratios."""
    return {
        depth: {
            "mean": float(np.mean(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
        }
        for depth, ratios in ratios_by_seed.items()
        if ratios
    }


def run_multiseed_benchmark(seeds: list[int], **benchmark_kwargs) -> dict:
    """Repeat a benchmark and summarize QAOA-guided approximation ratios."""
    ratios: dict[int, list[float]] = {}
    for seed in seeds:
        results = run_benchmark(seed=seed, **benchmark_kwargs)
        ground = results["E_ground"]
        if ground is None:
            continue
        for label, result in results["quantum_results"].items():
            if label.startswith("QAOA p="):
                depth = int(label.split("=")[1])
                ratios.setdefault(depth, []).append(result.best_energy / ground)

    summary = summarize_approximation_ratios(ratios)
    print("\nMulti-seed QAOA-guided approximation ratios:")
    for depth, values in sorted(summary.items()):
        print(
            f"  p={depth}: mean={values['mean']:.3f}, "
            f"range=[{values['min']:.3f}, {values['max']:.3f}]"
        )
    return summary


def exact_ground_energy(graph, n_nodes: int) -> float | None:
    """Return the brute-force Ising ground energy when the graph is small."""
    if n_nodes > 22:
        return None
    return min(
        ising_energy(graph, np.array([1 - 2 * ((bits >> i) & 1) for i in range(n_nodes)]))
        for bits in range(2**n_nodes)
    )


def select_backend(use_cloud: bool, shots: int):
    """Construct the backend requested by the benchmark configuration."""
    if use_cloud:
        from divi.backends import JobConfig, QoroService

        return QoroService(job_config=JobConfig(shots=shots)), "QoroService"
    return MaestroSimulator(shots=shots), "MaestroSimulator"


def format_result(label: str, result: ClusterAlgoResult, ground_energy, extra: str = "") -> str:
    """Format one benchmark result consistently across classical and quantum runs."""
    line = f"  [{label:<24}] best E = {result.best_energy:7.1f}"
    if ground_energy is not None:
        mean_ratio = float(np.mean([energy / ground_energy for energy in result.energy_history]))
        line += f" | mean r = {mean_ratio:.3f} | best r = {result.best_energy / ground_energy:.3f}"
    return f"{line} | {extra}" if extra else line


def run_benchmark(
    n_nodes: int = 18,
    degree: int = 10,
    qaoa_depths: list[int] | None = None,
    pce_encodings: list[str] | None = None,
    use_qdrift: bool = False,
    n_iterations_factor: int = 100,
    n_repetitions: int = 20,
    lambda_scale: float = 6,
    seed: int = 42,
    use_cloud: bool = False,
    shots: int = 10_000,
    output_dir: str = ".",
):
    """Run the Quantum-Guided Cluster Algorithm benchmark.

    Args:
        n_nodes: Number of graph nodes.
        degree: Graph regularity. Use 10+ for hard instances.
        qaoa_depths: QAOA depths to compare. Each uses ``n_nodes`` qubits.
            ``None`` defaults to ``[1, 2, 3, 5]``; pass ``[]`` to skip QAOA.
        pce_encodings: PCE encodings (``"dense"``, ``"poly"``). Each compresses
            ``n_nodes`` variables into far fewer qubits. Defaults to ``[]``.
        use_qdrift: If True, every QAOA run uses QDrift trotterization —
            randomized Trotter sampling that produces shallower circuits at
            higher depths. Recommended for ``p ≥ 3``.
        n_iterations_factor: Total iterations = factor * n_nodes.
        n_repetitions: Number of random restarts per method.
        lambda_scale: Cluster formation scaling parameter.
        seed: Random seed.
        use_cloud: If True, use QoroService (for >18 qubits or larger PCE).
        shots: Number of measurement shots.
        output_dir: Directory for saving plots.
    """
    if qaoa_depths is None:
        qaoa_depths = [1, 2, 3, 5]
    if pce_encodings is None:
        pce_encodings = []
    os.makedirs(output_dir, exist_ok=True)

    G = generate_random_maxcut_graph(n_nodes, degree, seed=seed)
    print(f"Graph: {n_nodes} nodes, {degree}-regular, {G.number_of_edges()} edges. "
          f"Budget: {n_iterations_factor}×n iters × {n_repetitions} restarts.")

    E_ground = exact_ground_energy(G, n_nodes)
    if E_ground is not None:
        print(f"Exact ground state: E₀ = {E_ground:.1f}")

    backend, backend_name = select_backend(use_cloud, shots)
    print(f"Backend: {backend_name} (shots={shots})")

    # SA baseline.
    t0 = time.time()
    sa_result = simulated_annealing(
        G, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, seed=seed,
    )
    print(format_result("SA", sa_result, E_ground, extra=f"{time.time() - t0:.1f}s"))

    # Coupling-constant cluster.
    Z_cc = coupling_constant_correlations(G)
    t0 = time.time()
    cc_result = correlation_guided_cluster_algorithm(
        G, Z_cc, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, lambda_scale=1, seed=seed,
    )
    print(format_result("Cluster (J coupling)", cc_result, E_ground,
                  extra=f"accept={cc_result.acceptance_rate:.1%} | {time.time() - t0:.1f}s"))

    # Quantum-guided sources (QAOA + PCE share the same downstream pipeline).
    quantum_specs: list[tuple[str, dict]] = (
        [("qaoa", {"n_layers": p, "use_qdrift": use_qdrift}) for p in qaoa_depths]
        + [("pce", {"encoding": enc}) for enc in pce_encodings]
    )
    quantum_results: dict[str, ClusterAlgoResult] = {}
    quantum_correlations: dict[str, np.ndarray] = {"J (coupling\nconstants)": Z_cc}
    qaoa_instances: dict = {}  # int -> QAOA instance, used by the QWC efficiency plot.

    for kind, kwargs in quantum_specs:
        if kind == "qaoa":
            corr = extract_qaoa_correlations(
                G, max_iterations=10, shots=shots, backend=backend, **kwargs
            )
            qaoa_instances[kwargs["n_layers"]] = corr.instance
        else:
            corr = extract_pce_correlations(
                G, shots=shots, backend=backend, **kwargs
            )
        quantum_correlations[f"{corr.label}\n({corr.n_qubits} qubits)"] = corr.Z

        t0 = time.time()
        cluster_result = correlation_guided_cluster_algorithm(
            G, corr.Z, n_iterations_factor=n_iterations_factor,
            n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
        )
        quantum_results[corr.label] = cluster_result
        print(format_result(
            f"{corr.label}-Guided", cluster_result, E_ground,
            extra=(f"accept={cluster_result.acceptance_rate:.1%} | "
                   f"circuits={corr.total_circuit_count} | {time.time() - t0:.1f}s"),
        ))

    results = {
        "graph": G,
        "sa_result": sa_result,
        "cc_result": cc_result,
        "quantum_results": quantum_results,
        "E_ground": E_ground,
        "n_iterations_factor": n_iterations_factor,
    }

    plot_approximation_ratios(
        results, save_path=os.path.join(output_dir, "1_approximation_ratios.png")
    )
    plot_correlation_heatmaps(
        G, quantum_correlations,
        save_path=os.path.join(output_dir, "2_correlation_heatmaps.png"),
    )
    if qaoa_instances:
        plot_circuit_efficiency(
            qaoa_instances, G.number_of_edges(),
            save_path=os.path.join(output_dir, "3_circuit_efficiency.png"),
        )
    plot_energy_distribution(
        results, save_path=os.path.join(output_dir, "4_energy_distributions.png")
    )
    return results


if __name__ == "__main__":
    run_benchmark(
        n_nodes=16,
        degree=12,
        qaoa_depths=[1, 2, 3],
        # pce_encodings=["dense"],  # uncomment to add a PCE row to the comparison
        n_iterations_factor=500,
        n_repetitions=30,
        lambda_scale=4,
        seed=42,
        use_cloud=False,
        shots=10_000,
        output_dir="plots",
    )
