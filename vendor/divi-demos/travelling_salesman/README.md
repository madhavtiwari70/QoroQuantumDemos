# Travelling Salesman Problem with QUBO, QAOA, and PCE

This example finds a short cyclic tour through randomly placed cities. It presents three representations of the same optimization task:

1. Direct QAOA on the uncompressed QUBO.
2. Partitioned QAOA with `PartitioningProgramEnsemble` and beam-search aggregation.
3. PCE, which encodes QUBO variables through qubit correlations.

## Run locally

```bash
python travelling_salesman.py
```

Or open `travelling_salesman.ipynb` for the guided version.

`N_CITIES_SMALL` controls the direct-QAOA and PCE instance. `N_CITIES_LARGE` controls the partitioned-QAOA example. The classical brute-force solution is shown only as a small-instance reference.
