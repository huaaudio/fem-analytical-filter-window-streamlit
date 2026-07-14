from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.io import wavfile

from secondment.fem_analytical_filter_window_app import (
    ANALYTICAL_PANEL_AUDIO_LABEL,
    BARE_PANEL_AUDIO_LABEL,
    DEFAULT_RESONANCE_HZ,
    DIAGNOSTIC_AUDIO_LABELS,
    FEM_PANEL_AUDIO_LABEL,
    INFINITE_PANEL_AUDIO_LABEL,
    ORIGINAL_AUDIO_LABEL,
    PRIMARY_AUDIO_LABELS,
    UPLOAD_AUDIO_LABEL,
    build_demo_values,
    build_diagnostic_signals,
    build_level_deltas,
    build_listening_summary,
    compute_curves,
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
    original, bare, analytical, fem = PRIMARY_AUDIO_LABELS
    signals = {
        original: np.ones(128),
        bare: np.full(128, 0.75),
        analytical: np.full(128, 0.5),
        fem: np.full(128, 0.25),
    }

    deltas = build_level_deltas(signals)

    assert list(deltas) == [original, bare, analytical, fem]
    assert deltas[original] == pytest.approx(0.0)
    assert deltas[bare] == pytest.approx(-2.4988, abs=1e-3)
    assert deltas[analytical] == pytest.approx(-6.0206, abs=1e-3)
    assert deltas[fem] == pytest.approx(-12.0412, abs=1e-3)
    assert describe_level_delta(deltas[analytical]) == "6.0 dB quieter overall than the original"

    summary = build_listening_summary(deltas)
    assert "bare A4 panel" in summary
    assert "2.5 dB quieter" in summary
    assert "A4 analytical version" in summary
    assert "6.0 dB quieter" in summary
    assert "FEM version" in summary
    assert "12.0 dB quieter" in summary
    assert "playback continues" in summary


def test_summary_explains_when_fem_result_is_missing() -> None:
    original, bare, analytical, _ = PRIMARY_AUDIO_LABELS
    summary = build_listening_summary({original: 0.0, bare: -1.0, analytical: -3.2})
    assert "matching FEM result is not available" in summary


def test_level_deltas_require_an_explicit_original_reference() -> None:
    with pytest.raises(ValueError, match="Original reference"):
        build_level_deltas({"A model": np.ones(8)})


def test_diagnostic_tracks_preserve_original_four_track_order() -> None:
    primary = {
        ORIGINAL_AUDIO_LABEL: np.ones(4),
        BARE_PANEL_AUDIO_LABEL: np.full(4, 0.8),
        ANALYTICAL_PANEL_AUDIO_LABEL: np.full(4, 0.6),
        FEM_PANEL_AUDIO_LABEL: np.full(4, 0.4),
    }
    diagnostics = build_diagnostic_signals(primary, np.full(4, 0.5))
    assert tuple(diagnostics) == DIAGNOSTIC_AUDIO_LABELS
    assert BARE_PANEL_AUDIO_LABEL not in diagnostics
    assert np.array_equal(diagnostics[INFINITE_PANEL_AUDIO_LABEL], np.full(4, 0.5))


def test_compute_curves_returns_distinct_bare_a4_baseline() -> None:
    values = build_demo_values(420.0)
    values.pop("show_fem_cache")
    values.pop("show_fem_lrm")
    _infinite, resonant, bare = compute_curves(**values)

    assert tuple(resonant) == ("A4",)
    assert tuple(bare) == ("A4",)
    target_index = int(np.argmin(np.abs(resonant["A4"].freqs_hz - 420.0)))
    assert not np.isclose(
        resonant["A4"].stl_db[target_index],
        bare["A4"].stl_db[target_index],
    )


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
