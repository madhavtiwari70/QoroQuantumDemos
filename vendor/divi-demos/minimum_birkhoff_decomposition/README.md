# Minimum Birkhoff Decomposition

Birkhoff decomposition expresses a doubly stochastic matrix as a convex combination of permutation matrices. This example connects raw circuit-sampling results to a classical CPLEX integer program.

It demonstrates a standalone Divi `CircuitPipeline`: a parameterized circuit is measured as shot counts, `cost_fn(params)` decodes those counts, and a Divi optimizer minimizes the resulting classical loss.

## Installation and run

```bash
pip install -r requirements.txt
python main.py
```

For an interactive walkthrough, open `birkhoff_decomposition.ipynb`.

## Useful commands

```bash
python main.py -n 4 -k 2 -inst 5 -it 20
python main.py --help
```

`birkhoff.py` contains the reusable `run_birkhoff(...)` orchestration; `main.py` loads an instance and displays the reconstruction.
