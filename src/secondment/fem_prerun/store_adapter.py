from __future__ import annotations

import numpy as np

from .lookup import SOLVER_ID, fem_lookup_signature


def _numerical_store():
    try:
        from secondment import numerical_store
    except ImportError:
        import numerical_store  # type: ignore
    return numerical_store


def tl_db_to_tau(tl_db) -> np.ndarray:
    return np.power(10.0, -np.asarray(tl_db, dtype=np.float64) / 10.0)


def build_signature(variant, base_dat_path, base_dat_hash):
    return fem_lookup_signature(
        base_dat_path,
        base_dat_hash,
        variant.f_res_hz,
        variant.m_ratio,
        variant.eta_res,
    )


def ingest_curve(store, freqs, tl_db, variant, base_dat_path, base_dat_hash) -> str:
    numerical_store = _numerical_store()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    tl_arr = np.asarray(tl_db, dtype=np.float64)
    signature, numerical_inputs = build_signature(variant, base_dat_path, base_dat_hash)
    normalized_result = {
        "solver_id": SOLVER_ID,
        "freqs": freqs_arr,
        "tau_meta_numerical": tl_db_to_tau(tl_arr),
        "tl_meta_numerical": tl_arr,
        "details": {"variant_id": variant.variant_id},
    }
    numerical_store.store_numerical_result(
        store,
        job_signature=signature,
        numerical_inputs=numerical_inputs,
        normalized_result=normalized_result,
    )
    return signature


def purge_fem_entries(store) -> int:
    removed = 0
    for signature in store.list_numerical_signatures():
        record = store.get_numerical_result(signature)
        if record is not None and record.metadata.get("solver_id") == SOLVER_ID:
            if store.delete_numerical_result(signature):
                removed += 1
    return removed


def find_matching_fem_entry(
    store,
    *,
    f_res_hz,
    m_ratio,
    eta_res,
    base_dat_hash=None,
    atol=1e-9,
) -> tuple[str, object] | None:
    """Find a FEM cache entry by portable physical inputs.

    The canonical job signature includes the resolved deck path. That is useful
    within one machine, but a store copied back from a remote server may carry a
    different absolute path. Matching by solver id, deck hash, and variant inputs
    keeps copied stores usable without re-ingesting locally.
    """
    expected_model_identity = None
    if base_dat_hash is not None:
        expected_model_identity = f"sha256:{base_dat_hash}"

    for signature in store.list_numerical_signatures():
        record = store.get_numerical_result(signature)
        if record is None or record.metadata.get("solver_id") != SOLVER_ID:
            continue
        inputs = dict(record.metadata.get("numerical_inputs") or {})
        if expected_model_identity is not None and inputs.get("model_identity") != expected_model_identity:
            continue
        try:
            same_inputs = (
                np.isclose(float(inputs.get("f_res_hz")), float(f_res_hz), rtol=0.0, atol=atol)
                and np.isclose(float(inputs.get("m_ratio")), float(m_ratio), rtol=0.0, atol=atol)
                and np.isclose(float(inputs.get("eta_res")), float(eta_res), rtol=0.0, atol=atol)
            )
        except (TypeError, ValueError):
            continue
        if same_inputs:
            return str(signature), record
    return None
