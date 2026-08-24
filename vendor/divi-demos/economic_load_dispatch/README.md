# Economic Load Dispatch with Prohibited Operating Zones

This example minimizes fuel cost while meeting demand and avoiding generator operating ranges that are not permitted.

It demonstrates how to encode a constrained dispatch problem as a QUBO, solve it with PCE, decode the sampled assignments, and repair them into feasible dispatches.

## Run locally

```bash
python economic_load_dispatch.py
```

The notebook provides the same walkthrough:

```bash
jupyter notebook economic_load_dispatch.ipynb
```

## Concepts

- Four binary variables encode the output of each generator.
- PCE maps the QUBO to a smaller qubit representation.
- The repair step enforces demand and prohibited operating zones after sampling.
- A classical simulated-annealing result provides a reference point.

The script starts with a three-generator instance and also includes a six-generator variant for experimenting with the same formulation. Both can use a local simulator.
