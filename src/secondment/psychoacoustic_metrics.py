"""Native DIN sharpness and Aures tonality for listening comparisons.

The bundled C sources are extracted from the METAVISION secondment analysis
pipeline. A small shared library is compiled into the operating system's
temporary cache on first use, then loaded with ``ctypes``. This keeps the app
portable across Windows development and Linux Streamlit deployment without
depending on MoSQITo or committing platform-specific binaries.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


METRIC_SAMPLE_RATE = 48_000
EXCEEDED_PERCENT = 5.0
N_BARK_BANDS = 240
SOUND_FIELD_FREE = 0
METHOD_TIME_VARYING = 1
TONALITY_FRAME_SECONDS = 0.160
TONALITY_HOP_RATIO = 0.5

_NATIVE_SOURCE_DIR = Path(__file__).resolve().parent / "native" / "psychohelperc"
_NATIVE_SOURCE_NAMES = (
    "ISO_532-1.c",
    "ISO_532-1.h",
    "tonality_aures1985.c",
    "tonality_aures1985.h",
    "pocketfft.c",
    "pocketfft.h",
)
_NATIVE_COMPILE_NAMES = (
    "ISO_532-1.c",
    "tonality_aures1985.c",
    "pocketfft.c",
)
_NATIVE_BUILD_LOCK = threading.Lock()

_BARK = np.arange(1, N_BARK_BANDS + 1, dtype=np.float64) * 0.1
_SHARPNESS_WEIGHT = _BARK * np.where(
    _BARK <= 15.8,
    1.0,
    0.15 * np.exp(0.42 * (_BARK - 15.8)) + 0.85,
)


class NativePsychoacousticError(RuntimeError):
    """Raised when the bundled native metric backend cannot be used."""


@dataclass(frozen=True)
class TrackPsychoacoustics:
    """S5 sharpness and K5 tonality for one listening track."""

    sharpness_s5_acum: float
    tonality_k5_tu: float


class _InputData(ctypes.Structure):
    _fields_ = [
        ("NumSamples", ctypes.c_int),
        ("SampleRate", ctypes.c_double),
        ("pData", ctypes.POINTER(ctypes.c_double)),
    ]


_SpecificLoudnessPointers = ctypes.POINTER(ctypes.c_double) * N_BARK_BANDS


def value_exceeded(values: np.ndarray, percent: float = EXCEEDED_PERCENT) -> float:
    """Return the finite value exceeded by ``percent`` percent of frames."""

    quantile = 1.0 - float(percent) / 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("The exceeded percentage must be between 0 and 100.")
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, quantile))


def _native_source_digest() -> str:
    digest = hashlib.sha256()
    for name in _NATIVE_SOURCE_NAMES:
        source_path = _NATIVE_SOURCE_DIR / name
        if not source_path.is_file():
            raise NativePsychoacousticError(f"Missing native metric source: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(source_path.read_bytes())
    return digest.hexdigest()[:16]


def _library_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _compiler_path() -> str:
    candidates = ("gcc", "cc", "clang") if sys.platform == "win32" else ("cc", "gcc", "clang")
    for candidate in candidates:
        compiler = shutil.which(candidate)
        if compiler:
            return compiler
    raise NativePsychoacousticError(
        "A C compiler is required to build the bundled psychoacoustic backend."
    )


def _compile_native_library(target_path: Path) -> None:
    compiler = _compiler_path()
    suffix = _library_suffix()
    candidate_path = target_path.with_name(
        f".{target_path.stem}.{os.getpid()}{suffix}"
    )
    command = [compiler, "-O3", "-std=c99", "-DISO532_BUILD"]
    if sys.platform == "win32":
        command.extend(["-shared", "-static-libgcc", "-Wl,--export-all-symbols"])
    elif sys.platform == "darwin":
        command.extend(["-dynamiclib", "-fPIC"])
    else:
        command.extend(["-shared", "-fPIC"])
    command.extend(
        [
            "-o",
            str(candidate_path),
            *(str(_NATIVE_SOURCE_DIR / name) for name in _NATIVE_COMPILE_NAMES),
            "-lm",
        ]
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=_NATIVE_SOURCE_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        candidate_path.unlink(missing_ok=True)
        raise NativePsychoacousticError(
            "The bundled psychoacoustic backend could not be compiled."
        ) from exc

    if completed.returncode != 0 or not candidate_path.is_file():
        candidate_path.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix_detail = f" ({detail[-1]})" if detail else ""
        raise NativePsychoacousticError(
            f"The bundled psychoacoustic backend could not be compiled{suffix_detail}."
        )
    os.replace(candidate_path, target_path)


@lru_cache(maxsize=1)
def native_library_path() -> Path:
    """Build if needed and return the source-versioned shared-library path."""

    digest = _native_source_digest()
    suffix = _library_suffix()
    cache_dir = Path(tempfile.gettempdir()) / "metavision-psychoacoustics" / digest
    library_path = cache_dir / f"metavision_psychoacoustics{suffix}"
    if not library_path.is_file():
        with _NATIVE_BUILD_LOCK:
            if not library_path.is_file():
                _compile_native_library(library_path)
    return library_path


@lru_cache(maxsize=1)
def _native_library() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL(str(native_library_path()))
    except OSError as exc:
        raise NativePsychoacousticError(
            "The bundled psychoacoustic backend could not be loaded."
        ) from exc

    library.f_loudness_from_signal.argtypes = [
        ctypes.POINTER(_InputData),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        _SpecificLoudnessPointers,
        ctypes.c_int,
    ]
    library.f_loudness_from_signal.restype = ctypes.c_int
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library.tonality_aures1985.argtypes = [
        double_pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_int),
        double_pointer,
    ]
    library.tonality_aures1985.restype = ctypes.c_int
    return library


def _metric_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate mono audio and resample a metric-only copy to 48 kHz."""

    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("Psychoacoustic metrics require a mono signal.")
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("Psychoacoustic metrics require finite audio samples.")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("The audio sample rate must be positive.")
    if sample_rate != METRIC_SAMPLE_RATE:
        divisor = gcd(sample_rate, METRIC_SAMPLE_RATE)
        samples = resample_poly(
            samples,
            METRIC_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    return np.ascontiguousarray(samples, dtype=np.float64)


def _sharpness_s5(signal: np.ndarray, library: ctypes.CDLL) -> float:
    decimation = METRIC_SAMPLE_RATE // 2_000
    capacity = signal.size // decimation
    if capacity <= 0:
        raise ValueError("The signal is too short for DIN sharpness analysis.")

    loudness = np.empty(capacity, dtype=np.float64)
    specific_loudness = np.empty((N_BARK_BANDS, capacity), dtype=np.float64)
    specific_pointers = _SpecificLoudnessPointers(
        *(
            specific_loudness[index].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            for index in range(N_BARK_BANDS)
        )
    )
    input_data = _InputData(
        int(signal.size),
        float(METRIC_SAMPLE_RATE),
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    frames = library.f_loudness_from_signal(
        ctypes.byref(input_data),
        SOUND_FIELD_FREE,
        METHOD_TIME_VARYING,
        0.0,
        loudness.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        specific_pointers,
        capacity,
    )
    if frames < 0:
        raise NativePsychoacousticError(
            f"ISO 532-1 loudness returned error code {frames}."
        )
    if frames == 0:
        return 0.0

    specific = specific_loudness[:, :frames]
    total_loudness = np.sum(specific, axis=0)
    numerator = _SHARPNESS_WEIGHT @ specific
    sharpness = np.zeros(frames, dtype=np.float64)
    np.divide(
        0.11 * numerator,
        total_loudness,
        out=sharpness,
        where=total_loudness >= 1e-12,
    )
    return max(value_exceeded(sharpness), 0.0)


def _tonality_k5(signal: np.ndarray, library: ctypes.CDLL) -> float:
    window_length = int(round(METRIC_SAMPLE_RATE * TONALITY_FRAME_SECONDS))
    hop_length = int(round(window_length * TONALITY_HOP_RATIO))
    capacity = (signal.size - window_length) // hop_length
    if capacity <= 0:
        raise ValueError("The signal is too short for Aures tonality analysis.")

    tonality = np.empty(capacity, dtype=np.float64)
    frame_count = ctypes.c_int(capacity)
    statistics = np.empty(5, dtype=np.float64)
    result = library.tonality_aures1985(
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(signal.size),
        float(METRIC_SAMPLE_RATE),
        SOUND_FIELD_FREE,
        0.0,
        tonality.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        None,
        None,
        None,
        ctypes.byref(frame_count),
        statistics.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if result < 0:
        raise NativePsychoacousticError(
            f"Aures tonality returned error code {result}."
        )
    k5 = float(statistics[4])
    return max(k5, 0.0) if np.isfinite(k5) else 0.0


def compute_track_psychoacoustics(
    signal: np.ndarray,
    sample_rate: int,
) -> TrackPsychoacoustics:
    """Compute DIN sharpness S5 and Aures tonality K5 for one track."""

    metric_signal = _metric_signal(signal, sample_rate)
    library = _native_library()
    return TrackPsychoacoustics(
        sharpness_s5_acum=_sharpness_s5(metric_signal, library),
        tonality_k5_tu=_tonality_k5(metric_signal, library),
    )


def compute_psychoacoustic_comparison(
    signals: Mapping[str, np.ndarray],
    sample_rate: int,
) -> dict[str, TrackPsychoacoustics]:
    """Compute metrics sequentially while preserving listening-track order."""

    return {
        label: compute_track_psychoacoustics(signal, sample_rate)
        for label, signal in signals.items()
    }


__all__ = [
    "EXCEEDED_PERCENT",
    "METRIC_SAMPLE_RATE",
    "NativePsychoacousticError",
    "TrackPsychoacoustics",
    "compute_psychoacoustic_comparison",
    "compute_track_psychoacoustics",
    "native_library_path",
    "value_exceeded",
]
