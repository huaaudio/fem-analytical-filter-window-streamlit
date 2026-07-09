from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .response_record import build_response_record
from .response_store import ResponseStore


NUMERICAL_STORE_MODE_CACHE_PREFERRED = "cache_preferred"
NUMERICAL_STORE_MODE_PRECOMPUTED_ONLY = "precomputed_only"
VALID_NUMERICAL_STORE_MODES = {
    NUMERICAL_STORE_MODE_CACHE_PREFERRED,
    NUMERICAL_STORE_MODE_PRECOMPUTED_ONLY,
}
DEFAULT_NUMERICAL_STORE_RELATIVE_PATH = Path(".cache") / "numerical_responses.sqlite3"


def _to_json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def _response_to_tl_db(response: np.ndarray, response_kind: str) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    if response_kind == "stl_db":
        return values
    if response_kind == "power":
        return -10.0 * np.log10(np.maximum(values, 1e-12))
    if response_kind == "amplitude":
        return -20.0 * np.log10(np.maximum(np.abs(values), 1e-12))
    raise ValueError(f"Unsupported numerical response kind: {response_kind}")


def get_numerical_store_mode() -> str:
    configured_mode = os.getenv("SECONDMENT_NUMERICAL_MODE", NUMERICAL_STORE_MODE_CACHE_PREFERRED)
    if configured_mode not in VALID_NUMERICAL_STORE_MODES:
        return NUMERICAL_STORE_MODE_CACHE_PREFERRED
    return configured_mode


def get_numerical_store_path(project_root: str | Path) -> Path:
    override = os.getenv("SECONDMENT_NUMERICAL_STORE_DB")
    if override:
        return Path(override)
    return Path(project_root) / DEFAULT_NUMERICAL_STORE_RELATIVE_PATH


def build_numerical_response_record(job_signature: str, numerical_inputs: dict[str, Any], normalized_result: dict[str, Any]):
    details = _to_json_safe(normalized_result.get("details") or {})
    inputs = _to_json_safe(numerical_inputs)
    metadata = {
        "source_type": "numerical_solver",
        "solver_id": normalized_result["solver_id"],
        "numerical_signature": str(job_signature),
        "numerical_inputs": inputs,
        "model_identity": inputs.get("model_identity"),
        "details": details,
    }
    return build_response_record(
        freqs=normalized_result["freqs"],
        response=normalized_result["tau_meta_numerical"],
        response_kind="power",
        metadata=metadata,
    )


def store_numerical_result(
    store: ResponseStore,
    job_signature: str,
    numerical_inputs: dict[str, Any],
    normalized_result: dict[str, Any],
):
    record = build_numerical_response_record(job_signature, numerical_inputs, normalized_result)
    store.put_numerical_result(job_signature, record)
    return record


def load_numerical_result(store: ResponseStore, job_signature: str) -> dict[str, Any] | None:
    record = store.get_numerical_result(job_signature)
    if record is None:
        return None

    metadata = dict(record.metadata)
    return {
        "signature": str(metadata.get("numerical_signature", job_signature)),
        "inputs": dict(metadata.get("numerical_inputs") or {}),
        "solver_id": metadata.get("solver_id"),
        "freqs": np.asarray(record.freqs, dtype=np.float64),
        "tau_meta_numerical": np.asarray(record.response, dtype=np.float64),
        "tl_meta_numerical": _response_to_tl_db(record.response, record.response_kind),
        "details": metadata.get("details") or {},
    }