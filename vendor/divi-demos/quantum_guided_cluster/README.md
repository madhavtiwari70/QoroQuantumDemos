# Quantum-Guided Cluster Algorithm for Max-Cut

This implementation follows the Quantum-Guided Cluster Algorithm described in [arXiv:2508.10656](https://arxiv.org/abs/2508.10656). It uses quantum-derived two-point correlations to guide classical cluster moves for Max-Cut.

The tutorial compares simulated annealing, coupling-based clusters, and correlations extracted by QAOA. PCE can provide the same correlation input through a compressed variable encoding.

## Run locally

```bash
python main.py
```

Or open `quantum_guided_cluster.ipynb`.

## Parameters to explore

- `n_nodes` and `degree`: the random regular Max-Cut graph.
- `qaoa_depths`: QAOA depths for correlation extraction.
- `pce_encodings`: optional `dense` or `poly` PCE extractors.
- `lambda_scale`: the paper's cluster-link probability parameter.
