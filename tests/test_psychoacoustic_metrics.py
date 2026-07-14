from __future__ import annotations

import shutil

import numpy as np
import pytest

from secondment.psychoacoustic_metrics import (
    compute_psychoacoustic_comparison,
    compute_track_psychoacoustics,
    value_exceeded,
)


def test_value_exceeded_matches_research_percentile_convention() -> None:
    assert value_exceeded(np.array([1.0, 2.0, 3.0]), 5.0) == pytest.approx(2.9)
    assert value_exceeded(np.array([np.nan, np.inf]), 5.0) == 0.0
    with pytest.raises(ValueError, match="between 0 and 100"):
        value_exceeded(np.array([1.0]), 101.0)


@pytest.mark.skipif(
    not any(shutil.which(name) for name in ("gcc", "cc", "clang")),
    reason="A C compiler is required for the native backend smoke test.",
)
def test_native_metrics_detect_tone_and_preserve_track_order() -> None:
    sample_rate = 11_025
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    tone = 0.05 * np.sin(2.0 * np.pi * 1_000.0 * time)
    noise = 0.05 * np.random.default_rng(42).normal(size=time.size)

    tone_metrics = compute_track_psychoacoustics(tone, sample_rate)
    noise_metrics = compute_track_psychoacoustics(noise, sample_rate)

    assert tone_metrics.sharpness_s5_acum == pytest.approx(1.0, abs=0.15)
    assert tone_metrics.tonality_k5_tu > noise_metrics.tonality_k5_tu
    assert tone_metrics.tonality_k5_tu > 0.8

    comparison = compute_psychoacoustic_comparison(
        {"Tone": tone, "Noise": noise},
        sample_rate,
    )
    assert list(comparison) == ["Tone", "Noise"]
