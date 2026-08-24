"""
Visualization utilities for the Quantum-Guided Cluster Algorithm benchmark.

Generates publication-quality dark-themed plots:
  1. Approximation ratio bar chart
  2. Correlation heatmaps (coupling constants vs QAOA depths)
  3. Circuit efficiency (QWC grouping savings)
  4. Energy distribution violin plots
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Dark theme color palette
# ──────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0f0f1a",
    "panel": "#1a1a2e",
    "text": "#e0e0f0",
    "accent": "#00d4ff",
    "sa": "#ff6b6b",
    "cc": "#ffa502",
    "qaoa_1": "#7bed9f",
    "qaoa_2": "#70a1ff",
    "qaoa_3": "#a29bfe",
    "qaoa_5": "#fd79a8",
    "grid": "#2d2d44",
}
QAOA_COLORS = ["#7bed9f", "#70a1ff", "#a29bfe", "#fd79a8", "#dfe6e9"]


def setup_dark_style():
    """Apply a premium dark theme."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 12,
    })


def plot_approximation_ratios(results: dict, save_path: str = "approximation_ratios.png"):
    """Plot 1: Bar chart of mean approximation ratio ± std across methods.

    This is the headline plot — shows that QAOA-guided clusters achieve
    higher mean r under a tight compute budget.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    E_ground = results["E_ground"]
    if E_ground is None or abs(E_ground) < 1e-10:
        print("⚠️  Cannot plot approximation ratios without ground state energy.")
        return

    # Collect method data
    methods = []
    means = []
    stds = []
    colors = []
    edge_colors = []

    # SA
    sa_ratios = [e / E_ground for e in results["sa_result"].energy_history]
    methods.append("Simulated\nAnnealing")
    means.append(np.mean(sa_ratios))
    stds.append(np.std(sa_ratios))
    colors.append(COLORS["sa"])
    edge_colors.append(COLORS["sa"])

    # Coupling Constants
    cc_ratios = [e / E_ground for e in results["cc_result"].energy_history]
    methods.append("Coupling\nConstants")
    means.append(np.mean(cc_ratios))
    stds.append(np.std(cc_ratios))
    colors.append(COLORS["cc"])
    edge_colors.append(COLORS["cc"])

    # Quantum-guided sources (QAOA at each depth, plus any PCE encodings)
    for i, (label, res) in enumerate(results["quantum_results"].items()):
        ratios = [e / E_ground for e in res.energy_history]
        methods.append(label.replace(" ", "\n"))
        means.append(np.mean(ratios))
        stds.append(np.std(ratios))
        c = QAOA_COLORS[i % len(QAOA_COLORS)]
        colors.append(c)
        edge_colors.append(c)

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, width=0.6,
                  color=[c + "40" for c in colors],
                  edgecolor=colors, linewidth=2,
                  capsize=5, error_kw={"elinewidth": 2, "capthick": 2,
                                       "ecolor": COLORS["text"]})

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=COLORS["text"])

    ax.set_ylabel("Approximation Ratio  r = E / E₀", fontsize=14, fontweight="bold")
    ax.set_title("Quantum-Guided Cluster Algorithm: Mean Approximation Ratio\n"
                 "(higher = better, error bars = ±1σ over random restarts)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(min(0.7, min(means) - 0.05), 1.02)
    ax.axhline(y=1.0, color=COLORS["accent"], linestyle="--", alpha=0.5, label="Optimal (r=1)")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_correlation_heatmaps(
    G: nx.Graph,
    correlations: dict[str, np.ndarray],
    save_path: str = "correlation_heatmaps.png",
):
    """Plot 2: Side-by-side heatmaps of the ZZ correlation matrix at each depth.

    Shows how QAOA correlations become more structured at higher depths,
    revealing the problem's spin-spin relationships.
    """
    setup_dark_style()
    n_plots = len(correlations)
    # constrained_layout handles the shared colorbar + suptitle correctly;
    # plt.tight_layout warns when colorbars span multiple axes.
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots + 1, 5), layout="constrained"
    )
    if n_plots == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    vmax = max(np.abs(Z).max() for Z in correlations.values())

    for ax, (label, Z) in zip(axes, correlations.items()):
        im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Qubit j", fontsize=11)
        ax.set_ylabel("Qubit i", fontsize=11)

    fig.suptitle("Two-Point Correlations ⟨ZᵢZⱼ⟩: Coupling Constants vs QAOA Depths",
                 fontsize=15, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("⟨ZᵢZⱼ⟩", fontsize=12)

    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_circuit_efficiency(
    qaoa_instances: dict[int, "QAOA"],
    n_edges: int,
    save_path: str = "circuit_efficiency.png",
):
    """Plot 3: Circuit count comparison showing QWC grouping efficiency.

    Demonstrates that Divi's QWC grouping reduces the number of circuits
    needed to measure all two-point correlations.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = sorted(qaoa_instances.keys())
    circuit_counts = [qaoa_instances[p].total_circuit_count for p in depths]

    # Naive baseline: one circuit per observable per iteration
    # For MAXCUT, there are n_edges ZZ terms in the Hamiltonian
    naive_per_iteration = n_edges

    x = np.arange(len(depths))
    width = 0.35

    ax.bar(x - width/2, circuit_counts, width,
           color=COLORS["accent"] + "60",
           edgecolor=COLORS["accent"], linewidth=2,
           label="Divi QWC Grouping (actual)")

    for i, count in enumerate(circuit_counts):
        ax.text(x[i] - width/2, count + 1, str(count),
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=COLORS["accent"])

    ax.axhline(y=n_edges, color=COLORS["sa"], linestyle="--", alpha=0.7,
               label=f"Observables in Hamiltonian ({n_edges})")

    ax.set_xlabel("QAOA Depth (p)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Circuits Executed", fontsize=14, fontweight="bold")
    ax.set_title("Circuit Efficiency: Divi's QWC Observable Grouping\n"
                 "groups commuting ZZ terms → fewer measurement circuits",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p = {p}" for p in depths], fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_energy_distribution(results: dict, save_path: str = "energy_distributions.png"):
    """Plot 4: Violin/box plot showing energy distribution across restarts.

    This reveals not just the best result but the reliability — QAOA-guided
    clusters should have a tighter distribution clustered near E₀.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    E_ground = results["E_ground"]
    if E_ground is None or abs(E_ground) < 1e-10:
        print("⚠️  Cannot plot distributions without ground state energy.")
        return

    # Collect all data
    all_data = []
    labels = []
    colors_list = []

    # SA
    sa_ratios = [e / E_ground for e in results["sa_result"].energy_history]
    all_data.append(sa_ratios)
    labels.append("Simulated\nAnnealing")
    colors_list.append(COLORS["sa"])

    # CC
    cc_ratios = [e / E_ground for e in results["cc_result"].energy_history]
    all_data.append(cc_ratios)
    labels.append("Coupling\nConstants")
    colors_list.append(COLORS["cc"])

    # Quantum-guided sources
    for i, (label, res) in enumerate(results["quantum_results"].items()):
        ratios = [e / E_ground for e in res.energy_history]
        all_data.append(ratios)
        labels.append(label.replace(" ", "\n"))
        colors_list.append(QAOA_COLORS[i % len(QAOA_COLORS)])

    positions = np.arange(len(all_data))
    parts = ax.violinplot(all_data, positions=positions, showmeans=True,
                          showmedians=True, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i] + "40")
        pc.set_edgecolor(colors_list[i])
        pc.set_linewidth(2)
        pc.set_alpha(0.7)

    for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color(COLORS["text"])
            parts[key].set_linewidth(1.5)

    ax.axhline(y=1.0, color=COLORS["accent"], linestyle="--", alpha=0.5,
               label="Optimal (r=1)")
    ax.set_ylabel("Approximation Ratio  r = E / E₀", fontsize=14, fontweight="bold")
    ax.set_title("Energy Distribution Across Random Restarts\n"
                 "(tighter around 1.0 = more reliable)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()
