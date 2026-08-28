"""
Minimum Birkhoff Decomposition — config-driven wrapper.

The original demo (main.py, copied unmodified from divi-demos/
minimum_birkhoff_decomposition/main.py) is already argument-driven via
argparse. This wrapper builds an args-like object from the data file and
calls its main() directly — no logic is duplicated. Terminal output
(including the matrix breakdown) is captured and returned as text.

Backend note: main() hardcodes `MaestroSimulator(shots=5000)` internally
and doesn't accept a backend parameter. To support switching to
QoroService from the data file without hand-editing the vendored file,
this wrapper temporarily substitutes what `MaestroSimulator` resolves to
inside that module before calling main() — Python looks up names at
call-time, so this safely redirects the call without modifying the
vendored source. The substitution is undone immediately after.
"""

import sys
import os
import io
import contextlib
from types import SimpleNamespace

VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "divi-demos", "minimum_birkhoff_decomposition")
sys.path.insert(0, os.path.abspath(VENDOR_DIR))

import main as bmo
from divi.backends import MaestroSimulator, QoroService, JobConfig


def run_from_config(cfg: dict, progress_callback=None) -> dict:
    args = SimpleNamespace(
        dim=cfg["dim"],
        comb=cfg["comb"],
        matrix_type=cfg["matrix_type"],
        instance=cfg["instance"],
        iterations=cfg["iterations"],
        optimizer=cfg["optimizer"],
    )

    b_cfg = cfg.get("backend", {"use_cloud": False, "shots": 5000})
    use_cloud = b_cfg.get("use_cloud", False)
    shots = b_cfg.get("shots", 5000)

    if progress_callback:
        backend_name = "QoroService" if use_cloud else "local MaestroSimulator"
        progress_callback(f"Running Birkhoff decomposition (n={args.dim}, k={args.comb}) on {backend_name}...")

    def backend_factory(shots=shots):  # noqa: shadow default matches main()'s call
        if use_cloud:
            return QoroService(job_config=JobConfig(shots=shots))
        return MaestroSimulator(shots=shots)

    original = bmo.MaestroSimulator
    bmo.MaestroSimulator = backend_factory
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bmo.main(args)
    finally:
        bmo.MaestroSimulator = original

    output_text = buf.getvalue()

    return {"output_text": output_text}
