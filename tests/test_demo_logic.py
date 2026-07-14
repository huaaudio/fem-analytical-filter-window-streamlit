from __future__ import annotations

import io
from types import SimpleNamespace

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
    build_model_snapshot_html,
    build_plot,
    build_psychoacoustic_figure,
    build_psychoacoustic_table,
    build_psychoacoustic_takeaway,
    compute_curves,
    describe_level_delta,
    interpolate_stl_at_frequency,
    load_requested_audio,
    preferred_resonance_frequency,
)
from secondment.psychoacoustic_metrics import TrackPsychoacoustics


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
    assert "bare panel" in summary
    assert "2.5 dB quieter" in summary
    assert "analytical version" in summary
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


def test_evidence_plot_omits_bare_panel_curve() -> None:
    infinite = SimpleNamespace(
        freqs_hz=np.array([100.0, 420.0, 1_000.0]),
        stl_db=np.array([10.0, 20.0, 30.0]),
    )
    resonant = {
        "A4": SimpleNamespace(
            freqs_hz=np.array([100.0, 420.0, 1_000.0]),
            stl_db=np.array([12.0, 35.0, 32.0]),
        )
    }

    figure = build_plot(infinite, resonant, 420.0)
    trace_names = [str(trace.name) for trace in figure.data]

    assert any("A4 metamaterial" in name for name in trace_names)
    assert any("infinite" in name.lower() for name in trace_names)
    assert not any("bare" in name.lower() for name in trace_names)


def test_psychoacoustic_comparison_preserves_order_and_separate_scales() -> None:
    metrics = {
        label: TrackPsychoacoustics(
            sharpness_s5_acum=1.0 + index / 10.0,
            tonality_k5_tu=0.2 + index / 10.0,
        )
        for index, label in enumerate(PRIMARY_AUDIO_LABELS)
    }

    table = build_psychoacoustic_table(metrics)

    assert table["Listening version"].tolist() == list(PRIMARY_AUDIO_LABELS)
    assert table["Sharpness S₅ (acum)"].tolist() == pytest.approx(
        [1.0, 1.1, 1.2, 1.3]
    )
    assert table["Tonality K₅ (t.u.)"].tolist() == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )

    figure = build_psychoacoustic_figure(metrics)
    assert len(figure.data) == 2
    assert all(trace.type == "bar" and trace.orientation == "h" for trace in figure.data)
    assert list(figure.data[0].x) == pytest.approx([1.3, 1.2, 1.1, 1.0])
    assert list(figure.data[1].x) == pytest.approx([0.5, 0.4, 0.3, 0.2])
    assert figure.layout.xaxis.range != figure.layout.xaxis2.range

    takeaway = build_psychoacoustic_takeaway(metrics)
    assert f"Highest sharpness: **{PRIMARY_AUDIO_LABELS[-1]}**" in takeaway
    assert f"Highest tonality: **{PRIMARY_AUDIO_LABELS[-1]}**" in takeaway


def test_model_snapshot_is_compact_and_omits_bare_prediction() -> None:
    snapshot = build_model_snapshot_html(420.0, 37.8, 42.0, 28.5)

    assert snapshot.count('class="mv-model-fact"') == 4
    assert "Target frequency" in snapshot
    assert "Metamaterial, analytical" in snapshot
    assert "Metamaterial, FEM" in snapshot
    assert "Infinite panel" in snapshot
    assert "Bare" not in snapshot


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
