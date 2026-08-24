# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Spectral modularity-based community detection for graph partitioning.

This module implements spectral methods for community detection based on modularity
optimization. The algorithms partition graphs into communities by maximizing modularity,
a measure of the quality of a network division. This is particularly useful for
breaking down large optimization problems (like portfolio optimization) into smaller,
more manageable sub-problems.

The main algorithm is based on the spectral optimization of modularity, where the
leading eigenvector of the modularity matrix is used to recursively split communities
until no further improvement in modularity can be achieved.

References:
    - Newman, M. E. J. (2006). "Modularity and community structure in networks."
      Proceedings of the National Academy of Sciences, 103(23), 8577-8582.
    - Newman, M. E. J. (2006). "Finding community structure in networks using the
      eigenvectors of matrices." Physical Review E, 74(3), 036104.
"""

import numpy as np
import numpy.typing as npt


def _modularity_matrix(
    adjacency_matrix: npt.NDArray[np.floating],
    degree_vector: npt.NDArray[np.floating],
    total_degree: float,
) -> npt.NDArray[np.floating]:
    """
    Compute the modularity matrix B for a given graph or subgraph.

    The modularity matrix measures the difference between the actual number of edges
    and the expected number of edges in a random graph with the same degree distribution.
    It is defined as:

        B_ij = A_ij - (k_i * k_j) / total_degree

    where:
        - A_ij is the adjacency matrix element (edge weight between nodes i and j)
        - k_i, k_j are the degrees of nodes i and j
        - total_degree is the sum of all edge weights (2m for undirected graphs)

    The modularity matrix is symmetric and has zero row/column sums. Its leading
    eigenvector is used in spectral methods to identify community structure.

    Args:
        adjacency_matrix: The adjacency matrix of the graph (can be weighted).
            Shape: (n, n) where n is the number of nodes.
        degree_vector: Vector of node degrees, where degree[i] is the sum of
            row/column i in the adjacency matrix. Shape: (n,).
        total_degree: Total sum of all edge weights in the graph. For undirected
            graphs, this equals 2m where m is the number of edges.

    Returns:
        The modularity matrix B with the same shape as adjacency_matrix.
        Returns a zero matrix if total_degree is zero (empty graph).

    Note:
        This function handles both the full graph and subgraphs. When used with
        subgraphs, the total_degree should typically be the global total_degree
        for consistency across the entire graph.
    """
    if total_degree == 0:
        # Handle empty graph case
        return np.zeros_like(adjacency_matrix)

    inv_total_degree = 1 / total_degree

    # Standard modularity formula: B_ij = A_ij - (k_i * k_j) / total_degree
    # This applies to both diagonal and off-diagonal elements
    B_hat_g = adjacency_matrix - inv_total_degree * np.outer(
        degree_vector, degree_vector
    )

    return B_hat_g


def _spectral_bisection(
    adjacency_matrix: npt.NDArray[np.floating],
    subgraph_indices: npt.NDArray[np.integer] | list[int],
    degree_vector: npt.NDArray[np.floating],
    total_degree: float,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | None:
    """
    Perform spectral bisection of a subgraph using the leading eigenvector.

    This is the core spectral bisection step used by both modularity optimization
    algorithms. It computes the modularity matrix for a subgraph, finds the leading
    eigenvector, and splits the subgraph based on the sign of eigenvector components.

    Args:
        adjacency_matrix: The full adjacency matrix of the graph. Shape: (n, n).
        subgraph_indices: Array or list of global node indices belonging to the subgraph.
        degree_vector: Vector of node degrees for the full graph. Shape: (n,).
        total_degree: Total degree to use for modularity matrix computation.
            Can be global (for optimization) or local (for threshold).

    Returns:
        A tuple (positive_subgraph, negative_subgraph) containing arrays of global
        node indices for the two partitions, or None if the subgraph cannot be split.

    Note:
        Returns None if:
        - Subgraph has < 2 nodes
        - Subgraph has no edges (total_degree == 0)
        - Leading eigenvector has all positive or all negative components
    """
    subgraph_indices = np.asarray(subgraph_indices)

    # Skip if subgraph is too small to split
    if len(subgraph_indices) < 2:
        return None

    # Extract subgraph adjacency and degrees
    sub_adj = adjacency_matrix[np.ix_(subgraph_indices, subgraph_indices)]
    sub_degrees = degree_vector[subgraph_indices]
    sub_total_degree = np.sum(sub_degrees)

    # Skip if subgraph has no edges
    if sub_total_degree == 0:
        return None

    # Compute modularity matrix
    B = _modularity_matrix(sub_adj, sub_degrees, total_degree)

    # Handle NaN/Inf values
    if np.isinf(B).any() or np.isnan(B).any():
        B = np.nan_to_num(
            B,
            copy=True,
            nan=0.0,
            posinf=len(subgraph_indices),
            neginf=-len(subgraph_indices),
        )

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    leading_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Split based on sign of eigenvector components
    positive_indices_local = np.where(leading_eigenvector >= 0)[0]
    negative_indices_local = np.where(leading_eigenvector < 0)[0]

    # Check if split is valid
    if len(positive_indices_local) == 0 or len(negative_indices_local) == 0:
        return None

    # Map local indices to global indices
    positive_subgraph = subgraph_indices[positive_indices_local]
    negative_subgraph = subgraph_indices[negative_indices_local]

    return positive_subgraph, negative_subgraph


def modularity_spectral_optimization(
    adjacency_matrix: npt.NDArray[np.floating],
) -> tuple[list[list[int]], npt.NDArray[np.integer]]:
    """
    Perform spectral modularity optimization to detect communities in a graph.

    This function implements a recursive spectral bisection algorithm that maximizes
    modularity by repeatedly splitting communities using the leading eigenvector of
    the modularity matrix. The algorithm:

    1. Starts with the entire graph as a single community
    2. For each community, computes the modularity matrix
    3. Finds the leading eigenvector (eigenvector corresponding to largest eigenvalue)
    4. Splits the community based on the sign of eigenvector components
    5. Only keeps splits that improve modularity (Qg > 0)
    6. Recursively processes sub-communities until no further improvement is possible

    The algorithm terminates when:
    - A subgraph is too small to split (< 2 nodes)
    - A subgraph has no edges (total_degree == 0)
    - The split would result in an empty partition
    - The modularity gain from splitting is non-positive (Qg <= 0)

    Args:
        adjacency_matrix: The adjacency matrix of the graph (can be weighted).
            Should be symmetric for undirected graphs. Shape: (n, n).

    Returns:
        A tuple containing:
        - communities: List of lists, where each inner list contains the
          node indices belonging to that community.
        - partition: A 1D array of integers where partition[i] is the
          community label for node i. Shape: (n,).

    Note:
        This is a private function that performs full modularity optimization.
        For threshold-based partitioning (stopping when communities reach a
        certain size), use `modularity_spectral_threshold` instead.

        If you only need the partition array, you can unpack with:
        ``_, partition = modularity_spectral_optimization(...)``

    Example:
        >>> import numpy as np
        >>> # Create a simple graph with two communities
        >>> adj = np.array([[0, 1, 1, 0, 0, 0],
        ...                 [1, 0, 1, 0, 0, 0],
        ...                 [1, 1, 0, 0, 0, 0],
        ...                 [0, 0, 0, 0, 1, 1],
        ...                 [0, 0, 0, 1, 0, 1],
        ...                 [0, 0, 0, 1, 1, 0]])
        >>> communities, partition = modularity_spectral_optimization(adj)
        >>> # Should identify two communities: [0, 1, 2] and [3, 4, 5]
    """
    n = len(adjacency_matrix)
    total_degree = np.sum(adjacency_matrix)
    degree_vector = np.sum(adjacency_matrix, axis=1)
    subgraphs = [np.arange(n)]

    partition = np.zeros(n, dtype=int)

    i = 0
    while subgraphs:
        current_subgraph = subgraphs.pop(0)

        # Perform spectral bisection
        result = _spectral_bisection(
            adjacency_matrix, current_subgraph, degree_vector, total_degree
        )
        if result is None:
            continue

        positive_subgraph, negative_subgraph = result

        # Update partition labels
        partition[positive_subgraph] = 2**i
        partition[negative_subgraph] = (2**i) + 1

        # Compute modularity for the split to check if it improves modularity
        sub_adj_pos = adjacency_matrix[np.ix_(positive_subgraph, positive_subgraph)]
        sub_adj_neg = adjacency_matrix[np.ix_(negative_subgraph, negative_subgraph)]
        sub_degrees_pos = degree_vector[positive_subgraph]
        sub_degrees_neg = degree_vector[negative_subgraph]

        B_pos = _modularity_matrix(sub_adj_pos, sub_degrees_pos, total_degree)
        B_neg = _modularity_matrix(sub_adj_neg, sub_degrees_neg, total_degree)

        # Calculate modularity gain
        Qg = (np.sum(B_pos) + np.sum(B_neg)) / total_degree

        i += 1

        # Only continue splitting if modularity gain is positive
        if Qg > 0:
            subgraphs.append(positive_subgraph)
            subgraphs.append(negative_subgraph)

    partition = np.unique(partition, return_inverse=True)[1]

    # Build communities list from partition
    communities = []
    for el in np.unique(partition):
        communities.append(list(np.where(partition == el)[0]))

    return communities, partition


##########################################################################


def modularity_spectral_threshold(
    adjacency_matrix: npt.NDArray[np.floating],
    threshold: int = 30,
) -> tuple[list[list[int]], npt.NDArray[np.integer]]:
    """
    Partition a graph into communities using spectral modularity with a size threshold.

    This function performs community detection using spectral modularity optimization,
    but stops splitting communities once they reach or fall below a specified size
    threshold. This is useful for applications like portfolio optimization where you
    want to break down large problems into smaller sub-problems of manageable size.

    The algorithm works by:
    1. Starting with the entire graph as a single community
    2. Recursively splitting communities using the leading eigenvector of the
       modularity matrix
    3. Stopping the split when a community size <= threshold
    4. Continuing to split larger communities until all are below the threshold

    Unlike `modularity_spectral_optimization`, this function prioritizes community
    size over modularity maximization, making it suitable for practical applications
    where problem size constraints are important.

    Args:
        adjacency_matrix: The adjacency matrix of the graph (can be weighted).
            Should be symmetric for undirected graphs. Shape: (n, n).
        threshold: Maximum size of communities. Communities with size <= threshold
            will not be further split. Default is 30.

    Returns:
        A tuple containing:
        - communities: List of lists, where each inner list contains the
          node indices belonging to that community. All communities will
          have size <= threshold (except possibly isolated nodes or communities
          that cannot be split further).
        - partition: A 1D array of integers where partition[i] is the
          community label for node i. Shape: (n,).

    Note:
        Communities may be larger than the threshold if:
        - They cannot be split (leading eigenvector has all positive or all
          negative components)
        - They have no edges (total_sub_degree == 0)
        - The split would result in an invalid partition

        If you only need the partition array, you can unpack with:
        ``_, partition = modularity_spectral_threshold(...)``

    Example:
        >>> import numpy as np
        >>> # Load a correlation matrix (e.g., from portfolio optimization)
        >>> correlation_matrix = np.load("correlation_matrix.npy")
        >>> # Partition into communities of at most 20 nodes
        >>> communities, partition = modularity_spectral_threshold(
        ...     correlation_matrix, threshold=20
        ... )
        >>> print(f"Found {len(communities)} communities")
        >>> print(f"Largest community size: {max(len(c) for c in communities)}")
        >>> # Or if you only need the partition:
        >>> _, partition = modularity_spectral_threshold(correlation_matrix, threshold=20)
    """

    def iterative_community_detection(initial_indices: list[int]) -> list[list[int]]:
        community_list = [initial_indices]
        degree_vector = np.sum(adjacency_matrix, axis=1)

        while community_list:
            subgraph_indices = community_list.pop()

            if len(subgraph_indices) <= threshold:
                yield subgraph_indices
                continue

            # Compute local total_degree for this subgraph
            sub_adj_matrix = adjacency_matrix[
                np.ix_(subgraph_indices, subgraph_indices)
            ]
            total_sub_degree = np.sum(sub_adj_matrix)

            # Perform spectral bisection using local total_degree
            result = _spectral_bisection(
                adjacency_matrix, subgraph_indices, degree_vector, total_sub_degree
            )

            if result is None:
                yield subgraph_indices
            else:
                positive_subgraph, negative_subgraph = result
                community_list.append(positive_subgraph.tolist())
                community_list.append(negative_subgraph.tolist())

    communities = list(
        iterative_community_detection(list(range(len(adjacency_matrix))))
    )

    # Create a partition map with community labels
    partition_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition_map[node] = i

    partition = np.array([partition_map[node] for node in range(len(adjacency_matrix))])

    return communities, partition
