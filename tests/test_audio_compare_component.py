from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from secondment.audio_compare_component import build_audio_comparison_payload
from secondment.fem_analytical_filter_window_app import PRIMARY_AUDIO_LABELS


def _decode_pcm16(data_uri: str) -> tuple[wave.Wave_read, np.ndarray]:
    prefix, encoded = data_uri.split(",", maxsplit=1)
    assert prefix == "data:audio/wav;base64"
    wav_file = wave.open(io.BytesIO(base64.b64decode(encoded)), "rb")
    samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return wav_file, samples


def test_payload_preserves_public_track_order_and_encodes_pcm16_wav() -> None:
    original, analytical, fem = PRIMARY_AUDIO_LABELS
    signals = {
        original: np.array([2.0, -2.0, 1.0, -1.0]),
        analytical: np.array([1.0, -1.0, 0.5, -0.5]),
        fem: np.array([0.5, -0.5, 0.25, -0.25]),
    }

    payload = build_audio_comparison_payload(
        signals,
        8_000,
        level_deltas_db={original: 0.0, analytical: -6.0, fem: -12.0},
    )

    assert payload["sampleRate"] == 8_000
    assert [track["label"] for track in payload["tracks"]] == [
        original,
        analytical,
        fem,
    ]
    assert [track["kind"] for track in payload["tracks"]] == [
        "original",
        "analytical",
        "fem",
    ]

    decoded: list[np.ndarray] = []
    for track in payload["tracks"]:
        wav_file, samples = _decode_pcm16(track["src"])
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 8_000
        assert wav_file.getnframes() == 4
        decoded.append(samples)
        wav_file.close()

    # One shared peak normalization preserves the intended 1 : 0.5 : 0.25 ratios.
    assert np.max(np.abs(decoded[0])) == 32_767
    assert np.max(np.abs(decoded[1])) == pytest.approx(16_384, abs=1)
    assert np.max(np.abs(decoded[2])) == pytest.approx(8_192, abs=1)


def test_payload_sanitizes_nonfinite_samples_before_encoding() -> None:
    payload = build_audio_comparison_payload(
        {"Original": np.array([np.nan, np.inf, -np.inf, 0.0])},
        48_000,
    )
    wav_file, samples = _decode_pcm16(payload["tracks"][0]["src"])
    wav_file.close()
    assert samples.tolist() == [0, 32_767, -32_767, 0]


@pytest.mark.parametrize("sample_rate", [0, -1, 8_000.5, True])
def test_payload_rejects_invalid_sample_rates(sample_rate: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_audio_comparison_payload({"Original": np.zeros(4)}, sample_rate)  # type: ignore[arg-type]
