"""
Visualization utilities for portfolio optimization.

This module provides functions for visualizing correlation matrices,
partition structures, and partition quality metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from modularity_spectral_partitioning import modularity_spectral_threshold


def _configure_sweep_axis(
    ax: Axes,
    thresholds: list[int],
    x_labels: list[str],
    ylabel: str,
    title: str,
) -> None:
    """
    Configure a matplotlib axis for sweep plots with common settings.

    Args:
        ax: Matplotlib axis to configure
        thresholds: List of threshold values
        x_labels: Labels for x-axis ticks
        ylabel: Label for y-axis
        title: Title for the plot
    """
    ax.set_xticks(thresholds)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=0)
    ax.set_xlabel("Threshold (n clusters)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def _reorder_by_partitions(
    partitions: npt.NDArray[np.integer],
    matrix: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], list[int]]:
    """
    Reorder matrix by grouping assets by partition.

    Args:
        partitions: Array mapping each asset to a partition ID
        matrix: Matrix to reorder (e.g., correlation matrix)

    Returns:
        Tuple of (reordered_matrix, partition_sizes)
    """
    unique_partitions = np.unique(partitions)
    partition_sizes = [np.sum(partitions == p) for p in unique_partitions]

    reorder_indices = []
    for partition_id in unique_partitions:
        partition_indices = np.where(partitions == partition_id)[0]
        reorder_indices.extend(partition_indices)

    return matrix[np.ix_(reorder_indices, reorder_indices)], partition_sizes


def _draw_partition_boundaries(ax: Axes, partition_sizes: list[int]) -> None:
    """
    Draw dashed lines between partitions on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        partition_sizes: List of sizes for each partition
    """
    cumulative_size = 0
    for i, size in enumerate(partition_sizes):
        if i > 0:
            ax.axhline(
                y=cumulative_size - 0.5,
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )
            ax.axvline(
                x=cumulative_size - 0.5,
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )
        cumulative_size += size


def _normalize_cluster_blocks(
    reordered_correlation: npt.NDArray[np.floating],
    partition_sizes: list[int],
) -> npt.NDArray[np.floating]:
    """
    Normalize each intra-cluster block independently.

    Each cluster block is normalized by its own maximum absolute value,
    preserving the sign structure. Inter-cluster correlations are set to zero.

    Args:
        reordered_correlation: Correlation matrix reordered by partitions
        partition_sizes: List of sizes for each partition

    Returns:
        Normalized correlation matrix with inter-cluster correlations set to zero
    """
    normalized_correlation = np.zeros_like(reordered_correlation)
    cumulative_size = 0

    for size in partition_sizes:
        start_idx, end_idx = cumulative_size, cumulative_size + size
        block = reordered_correlation[start_idx:end_idx, start_idx:end_idx].copy()

        # Remove diagonal for normalization, then restore
        np.fill_diagonal(block, 0)
        block_max_abs = np.max(np.abs(block))

        if block_max_abs > 0:
            normalized_block = block / block_max_abs
            np.fill_diagonal(normalized_block, 1.0)
            normalized_correlation[start_idx:end_idx, start_idx:end_idx] = (
                normalized_block
            )
        else:
            np.fill_diagonal(
                normalized_correlation[start_idx:end_idx, start_idx:end_idx], 1.0
            )

        cumulative_size += size

    return normalized_correlation


def plot_reordered_correlation(
    correlation_matrix: npt.NDArray[np.floating],
    partitions: npt.NDArray[np.integer],
) -> None:
    """
    Plot correlation matrix reordered by partitions.

    Args:
        correlation_matrix: Full correlation matrix
        partitions: Array mapping each asset to a partition ID
    """
    # Reorder matrix
    reordered_correlation, partition_sizes = _reorder_by_partitions(
        partitions, correlation_matrix
    )

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(
        reordered_correlation, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)
    _draw_partition_boundaries(ax, partition_sizes)
    ax.set_title(
        "Correlation Matrix Reordered by Partitions", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Asset Index (Reordered)", fontsize=10)
    ax.set_ylabel("Asset Index (Reordered)", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_partition_counts(
    correlation_matrix: npt.NDArray[np.floating],
    partitions: npt.NDArray[np.integer],
    threshold: int | None = None,
) -> None:
    """
    Plot intra-cluster correlation distributions and partition sizes.

    Args:
        correlation_matrix: Full correlation matrix
        partitions: Array mapping each asset to a partition ID
        threshold: Optional threshold line to draw on partition sizes plot
    """
    unique_partitions = np.unique(partitions)
    partition_sizes = [np.sum(partitions == p) for p in unique_partitions]

    # Extract intra-cluster correlations for each cluster
    intra_cluster_data = []
    for cluster_id in unique_partitions:
        cluster_indices = np.where(partitions == cluster_id)[0]
        cluster_block = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(cluster_block, k=1)
        intra_corrs = cluster_block[triu_indices]
        intra_cluster_data.append(intra_corrs)

    # Create 2-row, 1-column figure with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Common x-axis setup
    x_positions = range(len(partition_sizes))
    x_labels = [f"{pid}" for pid in unique_partitions]

    # Plot 1: Intra-Cluster Correlation Distributions (top)
    bp = ax1.boxplot(intra_cluster_data, positions=x_positions, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Correlation", fontsize=10)
    ax1.set_title(
        "Intra-Cluster Correlation Distributions", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])

    # Tighten y-axis based on actual data range
    all_corrs = (
        np.concatenate(intra_cluster_data) if intra_cluster_data else np.array([0])
    )
    y_min = max(-1, np.min(all_corrs) - 0.1)
    y_max = min(1, np.max(all_corrs) + 0.1)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Partition Sizes (bottom)
    ax2.bar(x_positions, partition_sizes)
    ax2.set_xlabel("Cluster ID", fontsize=10)
    ax2.set_ylabel("Number of Assets", fontsize=10)
    ax2.set_title("Partition Sizes", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, fontsize=8)
    # Format y-axis to show integers
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
    if threshold is not None:
        ax2.axhline(
            y=threshold, color="r", linestyle="dotted", label=f"Threshold: {threshold}"
        )
        ax2.legend()

    plt.tight_layout()
    plt.show()


def analyze_lambda_selection(
    covariance_matrix: npt.NDArray[np.floating],
    returns: npt.NDArray[np.floating],
) -> None:
    """
    Analyze risk-return scale to guide lambda parameter selection for QUBO.

    Args:
        covariance_matrix: Scaled covariance matrix
        returns: Scaled returns vector
    """
    typical_variance = np.mean(np.diag(covariance_matrix))
    typical_return = np.mean(np.abs(returns))

    base_ratio = typical_variance / typical_return if typical_return > 0 else 0

    print("=== Lambda Selection Guide ===")
    print(f"Risk/Return ratio: {base_ratio:.2f}")
    print()
    print("Suggested λ values:")
    print(f"  Conservative: {0.5 * base_ratio:.2f}")
    print(f"  Balanced:     {base_ratio:.2f}")
    print(f"  Aggressive:   {2.0 * base_ratio:.2f}")


def evaluate_partition_quality(
    correlation_matrix: npt.NDArray[np.floating],
    partitions: npt.NDArray[np.integer],
) -> dict[str, float | int | list[int]]:
    """
    Evaluate the quality of a partition by computing several metrics.

    Returns metrics that indicate:
    - How well assets within clusters correlate (should be high)
    - How independent clusters are from each other (should be low inter-cluster correlation)
    - How balanced the partition sizes are

    Args:
        correlation_matrix: Full correlation matrix
        partitions: Array mapping each asset to a partition ID

    Returns:
        Dictionary containing quality metrics
    """
    unique_partitions = np.unique(partitions)
    n_clusters = len(unique_partitions)

    # Compute intra-cluster and inter-cluster correlations
    intra_cluster_corrs = []
    inter_cluster_corrs = []
    cluster_sizes = []

    for i, cluster_id in enumerate(unique_partitions):
        cluster_indices = np.where(partitions == cluster_id)[0]
        cluster_size = len(cluster_indices)
        cluster_sizes.append(cluster_size)

        # Intra-cluster correlations (within this cluster)
        cluster_block = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(cluster_block, k=1)
        intra_corrs = cluster_block[triu_indices]
        intra_cluster_corrs.extend(intra_corrs)

        # Inter-cluster correlations (between this cluster and others)
        for j, other_cluster_id in enumerate(unique_partitions):
            if i < j:  # Only compute once per pair
                other_indices = np.where(partitions == other_cluster_id)[0]
                inter_block = correlation_matrix[np.ix_(cluster_indices, other_indices)]
                inter_cluster_corrs.extend(inter_block.flatten())

    # Compute statistics
    avg_intra = np.mean(intra_cluster_corrs) if intra_cluster_corrs else 0
    std_intra = np.std(intra_cluster_corrs) if intra_cluster_corrs else 0
    avg_inter = np.mean(inter_cluster_corrs) if inter_cluster_corrs else 0
    std_inter = np.std(inter_cluster_corrs) if inter_cluster_corrs else 0

    # Ratio of intra to inter (higher is better - shows clusters are well-separated)
    separation_ratio = avg_intra / abs(avg_inter) if avg_inter != 0 else float("inf")

    # Cluster size statistics (for balance)
    size_mean = np.mean(cluster_sizes)
    size_std = np.std(cluster_sizes)
    size_cv = size_std / size_mean if size_mean > 0 else 0  # Coefficient of variation

    # Modularity (approximate - using correlation as edge weights)
    # Modularity = (actual edges - expected edges) / total_edges
    total_degree = np.sum(np.abs(correlation_matrix))
    degree_vector = np.sum(np.abs(correlation_matrix), axis=1)

    modularity = 0.0
    for cluster_id in unique_partitions:
        cluster_indices = np.where(partitions == cluster_id)[0]
        cluster_adj = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
        cluster_degree = np.sum(np.abs(cluster_adj))
        expected_degree = (
            np.sum(degree_vector[cluster_indices]) ** 2 / total_degree
            if total_degree > 0
            else 0
        )
        modularity += cluster_degree - expected_degree
    modularity = modularity / total_degree if total_degree > 0 else 0

    return {
        "n_clusters": n_clusters,
        "avg_intra_cluster_corr": avg_intra,
        "std_intra_cluster_corr": std_intra,
        "avg_inter_cluster_corr": avg_inter,
        "std_inter_cluster_corr": std_inter,
        "separation_ratio": separation_ratio,
        "modularity": modularity,
        "cluster_size_mean": size_mean,
        "cluster_size_std": size_std,
        "cluster_size_cv": size_cv,  # Lower is better (more balanced)
        "cluster_sizes": cluster_sizes,
    }


def print_partition_quality(
    quality_metrics: dict[str, float | int | list[int]],
) -> None:
    """
    Print partition quality metrics in a formatted way.

    Args:
        quality_metrics: Dictionary returned by evaluate_partition_quality
    """
    print("=" * 70)
    print("PARTITION QUALITY METRICS")
    print("=" * 70)
    print(f"Number of Clusters: {quality_metrics['n_clusters']}")
    print()
    print("Correlation Metrics:")
    print(
        f"  Intra-Cluster ↑: {quality_metrics['avg_intra_cluster_corr']:.4f} ± {quality_metrics['std_intra_cluster_corr']:.4f}"
    )
    print(
        f"  Inter-Cluster ↓: {quality_metrics['avg_inter_cluster_corr']:.4f} ± {quality_metrics['std_inter_cluster_corr']:.4f}"
    )
    print(f"  Separation Ratio ↑: {quality_metrics['separation_ratio']:.2f}")
    print()
    print(f"Modularity ↑: {quality_metrics['modularity']:.4f}")
    print()
    print("Cluster Size Balance:")
    print(f"  Mean: {quality_metrics['cluster_size_mean']:.1f}")
    print(f"  Std Dev: {quality_metrics['cluster_size_std']:.2f}")
    print(f"  CV ↓: {quality_metrics['cluster_size_cv']:.3f}")


def sweep_partition_thresholds(
    correlation_matrix: npt.NDArray[np.floating],
    min_threshold: int = 10,
    max_threshold: int = 50,
    step: int = 5,
) -> dict[int, dict[str, float | int | list[int]]]:
    """
    Sweep over different partition size thresholds and evaluate quality for each.

    Args:
        correlation_matrix: Full correlation matrix
        min_threshold: Minimum partition size threshold to test
        max_threshold: Maximum partition size threshold to test
        step: Step size for threshold values

    Returns:
        Dictionary mapping threshold values to quality metrics dictionaries
    """
    results = {}
    thresholds = range(min_threshold, max_threshold + 1, step)

    for threshold in thresholds:
        _, partitions = modularity_spectral_threshold(
            correlation_matrix, threshold=threshold
        )
        quality_metrics = evaluate_partition_quality(correlation_matrix, partitions)
        results[threshold] = quality_metrics

    return results


def plot_partition_sweep_results(
    sweep_results: dict[int, dict[str, float | int | list[int]]],
) -> None:
    """
    Visualize how partition quality metrics change with threshold size.

    Args:
        sweep_results: Dictionary returned by sweep_partition_thresholds
    """
    thresholds = sorted(sweep_results.keys())
    metrics = {
        "modularity": [sweep_results[t]["modularity"] for t in thresholds],
        "separation_ratio": [sweep_results[t]["separation_ratio"] for t in thresholds],
        "n_clusters": [sweep_results[t]["n_clusters"] for t in thresholds],
        "avg_intra": [sweep_results[t]["avg_intra_cluster_corr"] for t in thresholds],
        "avg_inter": [sweep_results[t]["avg_inter_cluster_corr"] for t in thresholds],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_labels = [f"{t}\n({metrics['n_clusters'][i]})" for i, t in enumerate(thresholds)]

    # Plot 1: Modularity
    axes[0].plot(
        thresholds, metrics["modularity"], marker="o", linewidth=2, markersize=6
    )
    _configure_sweep_axis(
        axes[0], thresholds, x_labels, "Modularity", "Modularity vs Threshold ↑"
    )

    # Plot 2: Separation Ratio
    axes[1].plot(
        thresholds,
        metrics["separation_ratio"],
        marker="s",
        linewidth=2,
        markersize=6,
        color="green",
    )
    _configure_sweep_axis(
        axes[1],
        thresholds,
        x_labels,
        "Separation Ratio",
        "Separation Ratio vs Threshold ↑",
    )

    # Plot 3: Correlation Metrics
    axes[2].plot(
        thresholds,
        metrics["avg_intra"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Intra-cluster ↑",
        color="blue",
    )
    axes[2].plot(
        thresholds,
        metrics["avg_inter"],
        marker="s",
        linewidth=2,
        markersize=6,
        label="Inter-cluster ↓",
        color="red",
    )
    _configure_sweep_axis(
        axes[2],
        thresholds,
        x_labels,
        "Average Correlation",
        "Intra vs Inter-Cluster Correlation",
    )
    axes[2].legend()

    plt.suptitle(
        "Partition Quality Metrics vs Threshold Size",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    best_mod_idx = np.argmax(metrics["modularity"])
    best_sep_idx = np.argmax(metrics["separation_ratio"])
    print(
        f"\nBest Modularity: threshold={thresholds[best_mod_idx]}, modularity={metrics['modularity'][best_mod_idx]:.4f}, clusters={metrics['n_clusters'][best_mod_idx]}"
    )
    print(
        f"Best Separation: threshold={thresholds[best_sep_idx]}, ratio={metrics['separation_ratio'][best_sep_idx]:.2f}, clusters={metrics['n_clusters'][best_sep_idx]}"
    )
    print(
        f"\nRange: {min(metrics['n_clusters'])}-{max(metrics['n_clusters'])} clusters across thresholds"
    )
