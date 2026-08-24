# Portfolio Optimization with Quantum Algorithms

This notebook builds a Markowitz risk-return objective as a QUBO, partitions its correlation graph, and compares QAOA and PCE approaches.

It illustrates modularity-spectral partitioning, Divi's `PartitioningProgramEnsemble`, and beam-search aggregation of partition candidates. The supplied data and default notebook path use `MaestroSimulator` locally.

## Run

```bash
jupyter notebook portfolio_optimization.ipynb
```

Supporting modules provide the QUBO construction, partitioning routine, metrics, and visualization helpers.
