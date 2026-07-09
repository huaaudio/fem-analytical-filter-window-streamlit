"""Canonical FRF response records shared by uploads, numerical outputs, and persistence.

The core signal contract is `freqs + response + response_kind`. Source-specific provenance
belongs in `metadata`, which is intentionally restricted to JSON-serializable values so the
record can round-trip through the future persistent store without custom encoders.

Common metadata keys expected by downstream features include `source_type`, `display_name`,
`solver_id`, `numerical_signature`, `model_identity`, `details`, and optional precomputed
plotting values such as `stl_db`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_RESPONSE_KINDS = {"amplitude", "power", "stl_db"}


def _as_float64_vector(values, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"Response record {field_name} array must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Response record {field_name} array must contain only finite values.")
    array.setflags(write=False)
    return array


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a plain JSON-serializable metadata mapping for persistence-safe round-trips."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("Response record metadata must be a dictionary.")
    try:
        return json.loads(json.dumps(metadata, sort_keys=True))
    except TypeError as exc:
        raise ValueError(f"Response record metadata must be JSON-serializable: {exc}") from exc


def _compute_content_hash(freqs: np.ndarray, response: np.ndarray, response_kind: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(freqs, dtype=np.float64).tobytes())
    hasher.update(b"\x00")
    hasher.update(np.ascontiguousarray(response, dtype=np.float64).tobytes())
    hasher.update(b"\x00")
    hasher.update(str(response_kind).encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """Immutable canonical response record with persistence-safe metadata."""

    freqs: np.ndarray
    response: np.ndarray
    response_kind: str
    content_hash: str
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "freqs": self.freqs.tolist(),
            "response": self.response.tolist(),
            "response_kind": self.response_kind,
            "content_hash": self.content_hash,
            "metadata": _normalize_metadata(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ResponseRecord":
        record = build_response_record(
            freqs=payload["freqs"],
            response=payload["response"],
            response_kind=payload["response_kind"],
            metadata=payload.get("metadata", {}),
        )
        if payload.get("content_hash") != record.content_hash:
            raise ValueError("Response record payload content hash does not match canonical signal data.")
        return record


def build_response_record(
    freqs,
    response,
    response_kind: str,
    metadata: dict[str, Any] | None = None,
) -> ResponseRecord:
    canonical_freqs = _as_float64_vector(freqs, "freqs")
    canonical_response = _as_float64_vector(response, "response")

    if canonical_freqs.size != canonical_response.size:
        raise ValueError("Response record frequency and response arrays must have the same length.")
    if np.any(canonical_freqs < 0.0):
        raise ValueError("Response record frequencies must be non-negative.")
    if np.any(np.diff(canonical_freqs) <= 0.0):
        raise ValueError("Response record frequencies must be strictly increasing.")

    normalized_kind = str(response_kind)
    if not normalized_kind:
        raise ValueError("Response record kind must not be empty.")
    if normalized_kind not in SUPPORTED_RESPONSE_KINDS:
        raise ValueError(f"Unsupported response record kind: {normalized_kind}")

    normalized_metadata = _normalize_metadata(metadata)
    content_hash = _compute_content_hash(canonical_freqs, canonical_response, normalized_kind)

    return ResponseRecord(
        freqs=canonical_freqs,
        response=canonical_response,
        response_kind=normalized_kind,
        content_hash=content_hash,
        metadata=normalized_metadata,
    )