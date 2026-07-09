from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SOLVER_ID = "fem_sol108_diffuse"


def _read_deck_text(base_dat) -> str:
    with open(base_dat, "r", encoding="latin-1", newline="") as handle:
        return handle.read()


def parse_freq1_grid(base_dat) -> np.ndarray:
    """Derive the frequency grid from the first FREQ1 card in the deck."""
    for raw in _read_deck_text(base_dat).splitlines():
        if not raw.startswith("FREQ1"):
            continue
        tokens = raw.split()
        if len(tokens) < 5:
            raise ValueError(f"Malformed FREQ1 card: {raw!r}")
        f1 = float(tokens[2])
        df = float(tokens[3])
        ndf = int(float(tokens[4]))
        return f1 + df * np.arange(ndf + 1, dtype=np.float64)
    raise ValueError("No FREQ1 card found in base DAT for FEM frequency grid.")


def _build_numerical_job_signature(
    freqs,
    f_res,
    m_ratio,
    eta_res,
    mat_file_path,
    solver_id=SOLVER_ID,
    model_identity=None,
):
    resolved_model_identity = (
        str(model_identity) if model_identity is not None else str(Path(mat_file_path).resolve())
    )
    payload = {
        "solver_id": str(solver_id),
        "freqs_hz": np.asarray(freqs, dtype=np.float64).round(12).tolist(),
        "f_res_hz": round(float(f_res), 12),
        "m_ratio": round(float(m_ratio), 12),
        "eta_res": round(float(eta_res), 12),
        "mat_file_path": str(Path(mat_file_path).resolve()),
        "model_identity": resolved_model_identity,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return signature, payload


def fem_lookup_signature(base_dat, base_dat_hash, f_res, m_ratio, eta_res):
    """Canonical store signature for a precomputed FEM diffuse-TL curve."""
    grid = parse_freq1_grid(base_dat)
    return _build_numerical_job_signature(
        grid,
        f_res,
        m_ratio,
        eta_res,
        str(Path(base_dat).resolve()),
        solver_id=SOLVER_ID,
        model_identity=f"sha256:{base_dat_hash}",
    )
