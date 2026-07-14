from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.io import wavfile

from secondment.fem_analytical_filter_window_app import (
    DEFAULT_RESONANCE_HZ,
    PRIMARY_AUDIO_LABELS,
    UPLOAD_AUDIO_LABEL,
    build_level_deltas,
    build_listening_summary,
    describe_level_delta,
    interpolate_stl_at_frequency,
    load_requested_audio,
    preferred_resonance_frequency,
)


class UploadedBytes:
    """Small UploadedFile stand-in for pure audio validation tests."""

    def __init__(self, contents: bytes) -> None:
        self._contents = contents

    def getvalue(self) -> bytes:
        return self._contents


def _upload_request(contents: bytes) -> dict[str, object]:
    return {
        "source_type": UPLOAD_AUDIO_LABEL,
        "uploaded_file": UploadedBytes(contents),
        "duration_seconds": 8.0,
    }


def test_default_resonance_prefers_exact_420_hz() -> None:
    assert preferred_resonance_frequency([300.0, 410.0, 420.0, 430.0]) == 420.0
    assert preferred_resonance_frequency([]) == DEFAULT_RESONANCE_HZ == 420.0


def test_level_deltas_and_plain_language_summary() -> None:
    original, analytical, fem = PRIMARY_AUDIO_LABELS
    signals = {
        original: np.ones(128),
        analytical: np.full(128, 0.5),
        fem: np.full(128, 0.25),
    }

    deltas = build_level_deltas(signals)

    assert list(deltas) == [original, analytical, fem]
    assert deltas[original] == pytest.approx(0.0)
    assert deltas[analytical] == pytest.approx(-6.0206, abs=1e-3)
    assert deltas[fem] == pytest.approx(-12.0412, abs=1e-3)
    assert describe_level_delta(deltas[analytical]) == "6.0 dB quieter overall than the original"

    summary = build_listening_summary(deltas)
    assert "A4 analytical version" in summary
    assert "6.0 dB quieter" in summary
    assert "FEM version" in summary
    assert "12.0 dB quieter" in summary
    assert "playback continues" in summary


def test_summary_explains_when_fem_result_is_missing() -> None:
    original, analytical, _ = PRIMARY_AUDIO_LABELS
    summary = build_listening_summary({original: 0.0, analytical: -3.2})
    assert "matching FEM result is not available" in summary


def test_level_deltas_require_an_explicit_original_reference() -> None:
    with pytest.raises(ValueError, match="Original reference"):
        build_level_deltas({"A model": np.ones(8)})


@pytest.mark.parametrize(
    ("target_hz", "expected"),
    [(100.0, 10.0), (150.0, 15.0), (200.0, 20.0)],
)
def test_stl_interpolation_handles_endpoints_and_midpoint(
    target_hz: float, expected: float
) -> None:
    # Deliberately unordered input also verifies that the helper sorts by frequency.
    actual = interpolate_stl_at_frequency(
        np.array([200.0, 100.0]),
        np.array([20.0, 10.0]),
        target_hz,
    )
    assert actual == pytest.approx(expected)


def test_stl_interpolation_filters_invalid_points_and_rejects_extrapolation() -> None:
    assert interpolate_stl_at_frequency(
        np.array([100.0, np.nan, 200.0]),
        np.array([10.0, 99.0, 20.0]),
        150.0,
    ) == pytest.approx(15.0)
    assert interpolate_stl_at_frequency(
        np.array([100.0, 200.0]), np.array([10.0, 20.0]), 99.0
    ) is None
    assert interpolate_stl_at_frequency(
        np.array([np.nan]), np.array([np.nan]), 150.0
    ) is None


def test_uploaded_non_wav_is_rejected() -> None:
    with pytest.raises((ValueError, EOFError)):
        load_requested_audio(_upload_request(b"this is not a wav file"))


def test_uploaded_nonfinite_wav_is_rejected() -> None:
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, 8_000, np.array([0.0, np.nan, 0.25, -0.25], dtype=np.float32))

    with pytest.raises(ValueError, match="invalid sample values"):
        load_requested_audio(_upload_request(wav_buffer.getvalue()))
