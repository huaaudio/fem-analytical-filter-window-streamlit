from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from scipy.io import wavfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from secondment.analytical_filter import (
    AirCavity,
    HostPanel,
    LocalResonator,
    PartitionLeaf,
    bending_panel_pressure_filter,
    partition_pressure_filter,
)
from secondment.auralization_irbased import (
    auralize_with_frf,
    choose_processing_sample_rate,
    ensure_even_length,
    ensure_mono,
)
from secondment.compare_analytical_filter_window import (
    DEFAULT_PAPER_WINDOW_NAMES,
    PAPER_WINDOW_SIZES_M,
    paper_window_area_m2,
    paper_window_label,
)
from secondment.materials import get_material
from secondment.numerical_store import (
    get_numerical_store_path,
    load_numerical_result,
)
from secondment.response_store import ResponseStore
from secondment.fem_prerun.dat_writer import hash_base_dat
from secondment.fem_prerun.config import load_config as load_fem_prerun_config
from secondment.fem_prerun.lookup import SOLVER_ID as FEM_SOLVER_ID
from secondment.fem_prerun.lookup import fem_lookup_signature
from secondment.fem_prerun.store_adapter import find_matching_fem_entry


WINDOW_COLORS = {
    "A0": "#ff7f0e",
    "A1": "#2ca02c",
    "A2": "#d62728",
    "A3": "#9467bd",
    "A4": "#8c564b",
}
DEFAULT_AUDIO_SAMPLE_RATE = 48000
PROJECT_ROOT = SOURCE_ROOT.parent
FIXED_MASS_RATIO = 0.20
FIXED_RESONATOR_LOSS_FACTOR = 0.05
FIXED_UNIT_CELL_A_M = 0.05
FIXED_UNIT_CELL_B_M = 0.05
FIXED_DENSITY = 2700.0
FIXED_THICKNESS_M = 0.002
FIXED_YOUNG_MODULUS = 70e9
FIXED_POISSON_RATIO = 0.30
FIXED_HOST_LOSS_FACTOR = 0.02
FIXED_PARTITION_TYPE = "single"
DEFAULT_RESONANCE_HZ = 300.0

# openCFS FEM LRM (local-resonant metamaterial) STL exports, diffuse incidence.
FEM_RESULTS_DIR = PROJECT_ROOT / "fem" / "results"
_FEM_LRM_FILENAME_RE = re.compile(r"^(A\d)_wall_lrm\.csv$", re.IGNORECASE)
FEM_CACHE_CURVE_KEY = "Cache"
UPLOAD_AUDIO_LABEL = "Upload WAV"
RECORDED_AUDIO_CANDIDATES = (
    ("Heat pump measurement 10", "archive/Recordings/10_Measurement_10s.wav"),
    ("Heat pump measurement 12", "archive/Recordings/12_Measurement_10s.wav"),
    ("Heat pump max capacity 25", "archive/Recordings/25_MaxCapacity_10s.wav"),
    ("Heat pump defrost 28", "archive/Recordings/28_Defrost_10s.wav"),
    ("Car pass-by 0055", "archive/PA/repsig/car pass-by-0055.wav"),
    ("Motorbike idling 0009", "archive/PA/repsig/motorbike idling-0009.wav"),
    ("Vacuum cleaner 0024", "archive/PA/repsig/vacuum cleaner-0024.wav"),
    ("Airplanes short", "archive/Scripts_Auralization_Lucas_20260420/sounds/Airplanes-006_short.wav"),
    ("Air compressor", "archive/Scripts_Auralization_Lucas_20260420/sounds/air compressor-011.wav"),
    ("Jackhammer short", "archive/Scripts_Auralization_Lucas_20260420/sounds/jackhammer-001_short.wav"),
)


def read_fem_stl_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_hz, stl_db) from an Simcenter3D-exported STL CSV.

    The first two lines are ``#``-prefixed headers; data rows are ``freq,value``.
    Files carry a UTF-8 BOM, so they are read with ``utf-8-sig``.
    """
    freqs: list[float] = []
    stl: list[float] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            freq_str, _, val_str = line.partition(",")
            if not val_str:
                continue
            freqs.append(float(freq_str))
            stl.append(float(val_str))
    return np.asarray(freqs, dtype=np.float64), np.asarray(stl, dtype=np.float64)


# Compatibility alias for code that still uses the private name
_read_fem_stl_csv = read_fem_stl_csv


@st.cache_data(show_spinner=False)
def load_fem_lrm_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all FEM LRM STL curves keyed by paper window name (A0-A4)."""
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not FEM_RESULTS_DIR.is_dir():
        return curves
    for path in sorted(FEM_RESULTS_DIR.glob("*_wall_lrm.csv")):
        match = _FEM_LRM_FILENAME_RE.match(path.name)
        if match is None:
            continue
        window = match.group(1).upper()
        freqs, stl = read_fem_stl_csv(path)
        if freqs.size:
            curves[window] = (freqs, stl)
    return curves


def _open_numerical_store() -> ResponseStore:
    return ResponseStore(get_numerical_store_path(PROJECT_ROOT))


def prepare_fem_cache_lookup(config_path: Path | None = None) -> dict[str, object] | None:
    """Resolve the base DAT and hash used by the FEM prerun cache."""
    config_path = PROJECT_ROOT / "fem-prerun.json" if config_path is None else Path(config_path)
    if not config_path.exists():
        return None
    try:
        config = load_fem_prerun_config(config_path)
        base_dat = Path(config.base_dat)
        if not base_dat.exists():
            return None
        return {"base_dat": base_dat, "base_hash": hash_base_dat(base_dat)}
    except Exception:
        return None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_fem_cache_points(store: ResponseStore) -> list[dict[str, object]]:
    """List FEM SOL 108 cache points available in the shared numerical store."""
    points: list[dict[str, object]] = []
    for signature in store.list_numerical_signatures():
        record = store.get_numerical_result(signature)
        if record is None or record.metadata.get("solver_id") != FEM_SOLVER_ID:
            continue
        inputs = dict(record.metadata.get("numerical_inputs") or {})
        f_res_hz = _float_or_none(inputs.get("f_res_hz"))
        m_ratio = _float_or_none(inputs.get("m_ratio"))
        eta_res = _float_or_none(inputs.get("eta_res"))
        if f_res_hz is None or m_ratio is None or eta_res is None:
            continue
        details = dict(record.metadata.get("details") or {})
        points.append(
            {
                "signature": str(signature),
                "f_res_hz": f_res_hz,
                "m_ratio": m_ratio,
                "eta_res": eta_res,
                "variant_id": details.get("variant_id"),
                "n_freqs": int(record.freqs.size),
                "f_min_hz": float(record.freqs[0]),
                "f_max_hz": float(record.freqs[-1]),
            }
        )
    return sorted(points, key=lambda item: (item["m_ratio"], item["f_res_hz"], item["eta_res"]))


@st.cache_data(show_spinner=False)
def load_fem_cache_points_cached(store_path: str, mtime_ns: int, size: int) -> list[dict[str, object]]:
    del mtime_ns, size
    return collect_fem_cache_points(ResponseStore(Path(store_path)))


def load_fem_cache_points() -> tuple[list[dict[str, object]], str | None]:
    try:
        store_path = get_numerical_store_path(PROJECT_ROOT)
        if store_path.exists():
            stat = store_path.stat()
            return load_fem_cache_points_cached(str(store_path), stat.st_mtime_ns, stat.st_size), None
        return collect_fem_cache_points(ResponseStore(store_path)), None
    except Exception as exc:
        return [], f"FEM cache unavailable: {exc}"


def cached_resonance_frequencies(points: list[dict[str, object]]) -> list[float]:
    values = {
        float(point["f_res_hz"])
        for point in points
        if np.isclose(float(point["m_ratio"]), FIXED_MASS_RATIO, rtol=0.0, atol=1e-12)
        and np.isclose(
            float(point["eta_res"]),
            FIXED_RESONATOR_LOSS_FACTOR,
            rtol=0.0,
            atol=1e-12,
        )
    }
    return sorted(values)


def load_matching_fem_cache_curve(
    store: ResponseStore,
    fem_lookup: dict[str, object],
    resonance_hz: float,
    mass_ratio: float,
    resonator_loss_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    signature, _inputs = fem_lookup_signature(
        fem_lookup["base_dat"],
        fem_lookup["base_hash"],
        resonance_hz,
        mass_ratio,
        resonator_loss_factor,
    )
    result = load_numerical_result(store, signature)
    if result is None:
        fallback = find_matching_fem_entry(
            store,
            f_res_hz=resonance_hz,
            m_ratio=mass_ratio,
            eta_res=resonator_loss_factor,
            base_dat_hash=fem_lookup["base_hash"],
        )
        if fallback is not None:
            fallback_signature, _record = fallback
            result = load_numerical_result(store, fallback_signature)
    if result is None or result.get("solver_id") != FEM_SOLVER_ID:
        return None
    return (
        np.asarray(result["freqs"], dtype=np.float64),
        np.asarray(result["tl_meta_numerical"], dtype=np.float64),
    )


def load_current_fem_cache_curve(
    fem_lookup: dict[str, object] | None,
    resonance_hz: float,
    mass_ratio: float,
    resonator_loss_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if fem_lookup is None:
        return None
    try:
        return load_matching_fem_cache_curve(
            _open_numerical_store(),
            fem_lookup,
            resonance_hz,
            mass_ratio,
            resonator_loss_factor,
        )
    except Exception:
        return None


def fem_cache_point_matches(
    point: dict[str, object],
    resonance_hz: float,
    mass_ratio: float,
    resonator_loss_factor: float,
) -> bool:
    return (
        np.isclose(float(point["f_res_hz"]), float(resonance_hz), rtol=0.0, atol=1e-9)
        and np.isclose(float(point["m_ratio"]), float(mass_ratio), rtol=0.0, atol=1e-12)
        and np.isclose(float(point["eta_res"]), float(resonator_loss_factor), rtol=0.0, atol=1e-12)
    )


def nearest_fem_cache_point(
    points: list[dict[str, object]],
    resonance_hz: float,
    mass_ratio: float,
    resonator_loss_factor: float,
) -> dict[str, object] | None:
    if not points:
        return None
    f_values = np.asarray([float(point["f_res_hz"]) for point in points], dtype=np.float64)
    m_values = np.asarray([float(point["m_ratio"]) for point in points], dtype=np.float64)
    eta_values = np.asarray([float(point["eta_res"]) for point in points], dtype=np.float64)
    f_scale = max(float(np.ptp(f_values)), 1.0)
    m_scale = max(float(np.ptp(m_values)), 0.01)
    eta_scale = max(float(np.ptp(eta_values)), 0.001)

    def score(point: dict[str, object]) -> float:
        return (
            ((float(point["f_res_hz"]) - float(resonance_hz)) / f_scale) ** 2
            + ((float(point["m_ratio"]) - float(mass_ratio)) / m_scale) ** 2
            + ((float(point["eta_res"]) - float(resonator_loss_factor)) / eta_scale) ** 2
        )

    return min(points, key=score)


def build_fem_cache_points_table(points: list[dict[str, object]]) -> pd.DataFrame:
    rows = [
        {
            "f_res [Hz]": float(point["f_res_hz"]),
            "m_ratio [%]": 100.0 * float(point["m_ratio"]),
            "eta_res [-]": float(point["eta_res"]),
            "Freq range [Hz]": f"{float(point['f_min_hz']):.0f}-{float(point['f_max_hz']):.0f}",
            "N": int(point["n_freqs"]),
        }
        for point in points
    ]
    return pd.DataFrame(rows)


def discover_recorded_audio_files() -> dict[str, Path]:
    recordings: dict[str, Path] = {}
    for label, relative_path in RECORDED_AUDIO_CANDIDATES:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            recordings[label] = path
    return recordings


def fem_curve_display_label(key: str) -> str:
    if key == FEM_CACHE_CURVE_KEY:
        return "FEM cache (SOL 108, diffuse)"
    return f"FEM LRM {key} (CSV, diffuse)"


def fem_curve_color(key: str) -> str:
    if key == FEM_CACHE_CURVE_KEY:
        return "#111827"
    return WINDOW_COLORS.get(key, "#555555")


def build_frequency_axis(f_min_hz: float, f_max_hz: float, n_points: int) -> np.ndarray:
    if f_min_hz <= 0.0:
        raise ValueError("Minimum frequency must be positive.")
    if f_max_hz <= f_min_hz:
        raise ValueError("Maximum frequency must be greater than minimum frequency.")
    if n_points < 3:
        raise ValueError("Frequency points must be at least 3.")
    return np.geomspace(float(f_min_hz), float(f_max_hz), int(n_points))


@st.cache_data(show_spinner=False)
def compute_curves(
    f_min_hz: float,
    f_max_hz: float,
    n_points: int,
    selected_windows: tuple[str, ...],
    density: float,
    thickness_m: float,
    young_modulus: float,
    poisson_ratio: float,
    host_loss_factor: float,
    resonance_hz: float,
    mass_ratio: float,
    resonator_loss_factor: float,
    unit_cell_a_m: float,
    unit_cell_b_m: float,
    incidence: str,
    theta_oblique_deg: float,
    theta_limit_deg: float,
    theta_samples: int,
    radial_samples: int,
    material: str | None = None,
    partition_type: str = "single",
    cavity_thickness_m: float = 0.05,
    second_material: str | None = None,
    second_thickness_m: float | None = None,
    second_plate_type: str | None = None,
):
    freqs = build_frequency_axis(f_min_hz, f_max_hz, n_points)
    if material is not None:
        host = get_material(material).host_panel(thickness_m, loss_factor=host_loss_factor)
    else:
        host = HostPanel(
            density=density,
            thickness=thickness_m,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            loss_factor=host_loss_factor,
        )
    resonator = LocalResonator(
        resonance_frequency_hz=resonance_hz,
        mass_ratio=mass_ratio,
        unit_cell_area=unit_cell_a_m * unit_cell_b_m,
        stiffness_loss_factor=resonator_loss_factor,
    )
    # "normal" is a single plane wave at 0 deg, i.e. oblique incidence fixed to 0.
    effective_incidence = "oblique" if incidence == "normal" else incidence
    effective_theta_oblique_deg = 0.0 if incidence == "normal" else theta_oblique_deg
    common_kwargs = {
        "incidence": effective_incidence,
        "theta_oblique_rad": np.deg2rad(effective_theta_oblique_deg),
        "theta_limit_rad": np.deg2rad(theta_limit_deg),
        "theta_samples": theta_samples,
        "radial_samples": radial_samples,
    }

    if partition_type == "double":
        second_thickness = thickness_m if second_thickness_m is None else second_thickness_m
        second_name = material if second_material is None else second_material
        if second_name is not None:
            second_host = get_material(second_name).host_panel(
                second_thickness, loss_factor=host_loss_factor
            )
        else:
            second_host = HostPanel(
                density=density,
                thickness=second_thickness,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                loss_factor=host_loss_factor,
            )
        second_resonator = resonator if (second_plate_type or "meta") == "meta" else None
        leaves = (PartitionLeaf(host=host, resonator=resonator), PartitionLeaf(host=second_host, resonator=second_resonator))
        cavities = (AirCavity(thickness_m=cavity_thickness_m),)

        def filter_call(size_type: str, **window_kwargs):
            return partition_pressure_filter(
                freqs, leaves, cavities, size_type=size_type, **window_kwargs, **common_kwargs
            )
    else:
        def filter_call(size_type: str, **window_kwargs):
            return bending_panel_pressure_filter(
                freqs,
                host=host,
                resonator=resonator,
                size_type=size_type,
                **window_kwargs,
                **common_kwargs,
            )

    infinite = filter_call("infinite")
    finite = {
        paper_name: filter_call("finite", window_area_m2=paper_window_area_m2(paper_name))
        for paper_name in selected_windows
    }
    return infinite, finite


def build_plot(
    infinite,
    finite_results: dict[str, object],
    resonance_hz: float,
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> go.Figure:
    fig = go.Figure()
    freq_arrays = [np.asarray(infinite.freqs_hz, dtype=np.float64)]
    stl_arrays = [np.asarray(infinite.stl_db, dtype=np.float64)]
    fig.add_trace(
        go.Scatter(
            x=infinite.freqs_hz,
            y=infinite.stl_db,
            name="Infinite panel",
            mode="lines",
            line={"color": "#1f77b4", "width": 3},
        )
    )
    for paper_name, result in finite_results.items():
        freq_arrays.append(np.asarray(result.freqs_hz, dtype=np.float64))
        stl_arrays.append(np.asarray(result.stl_db, dtype=np.float64))
        fig.add_trace(
            go.Scatter(
                x=result.freqs_hz,
                y=result.stl_db,
                name=paper_window_label(paper_name),
                mode="lines",
                line={"color": WINDOW_COLORS.get(paper_name), "width": 2.5},
            )
        )
    for paper_name, (fem_freqs, fem_stl) in (fem_lrm_curves or {}).items():
        if fem_freqs.size:
            freq_arrays.append(np.asarray(fem_freqs, dtype=np.float64))
            stl_arrays.append(np.asarray(fem_stl, dtype=np.float64))
        fig.add_trace(
            go.Scatter(
                x=fem_freqs,
                y=fem_stl,
                name=fem_curve_display_label(paper_name),
                mode="markers",
                marker={
                    "color": fem_curve_color(paper_name),
                    "size": 6 if paper_name == FEM_CACHE_CURVE_KEY else 5,
                    "symbol": "diamond-open" if paper_name == FEM_CACHE_CURVE_KEY else "circle-open",
                },
            )
        )

    freq_values = np.concatenate([array[np.isfinite(array) & (array > 0.0)] for array in freq_arrays])
    stl_values = np.concatenate([array[np.isfinite(array)] for array in stl_arrays])
    f_min = float(np.min(freq_values))
    f_max = float(np.max(freq_values))
    y_min = float(np.min(stl_values))
    y_max = float(np.max(stl_values))
    y_pad = max(1.0, 0.05 * (y_max - y_min))
    y0 = y_min - y_pad
    y1 = y_max + y_pad
    fig.add_trace(
        go.Scatter(
            x=[resonance_hz, resonance_hz],
            y=[y0, y1],
            name="f_res",
            mode="lines",
            line={"color": "#202020", "width": 1, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[resonance_hz],
            y=[y1],
            text=["f_res"],
            mode="text",
            textposition="top center",
            textfont={"color": "#202020", "size": 12},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=560,
        margin={"l": 56, "r": 24, "t": 36, "b": 56},
        xaxis={
            "title": "Frequency [Hz]",
            "type": "log",
            "range": [float(np.log10(f_min)), float(np.log10(f_max))],
            "showgrid": True,
            "minor": {"showgrid": True},
        },
        yaxis={"title": "STL [dB]", "range": [y0, y1], "showgrid": True},
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
    )
    return fig


def build_summary_table(infinite, finite_results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for paper_name, result in finite_results.items():
        peak_index = int(np.argmax(result.stl_db))
        rows.append(
            {
                "Window": paper_name,
                "Size [m]": f"{PAPER_WINDOW_SIZES_M[paper_name][0]:.3f} x {PAPER_WINDOW_SIZES_M[paper_name][1]:.3f}",
                "Area [m2]": paper_window_area_m2(paper_name),
                "Peak STL [dB]": float(result.stl_db[peak_index]),
                "Peak frequency [Hz]": float(result.freqs_hz[peak_index]),
                "Mean gain vs infinite [dB]": float(np.mean(result.stl_db - infinite.stl_db)),
            }
        )
    return pd.DataFrame(rows)


def generate_audio_signal(
    source_type: str,
    duration_seconds: float,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    seed: int = 7,
) -> tuple[np.ndarray, int]:
    n_samples = max(2, int(round(float(duration_seconds) * int(sample_rate))))
    rng = np.random.default_rng(int(seed))
    if source_type == "Pink noise":
        white = rng.normal(0.0, 1.0, n_samples)
        spectrum = np.fft.rfft(white)
        spectrum = spectrum / np.sqrt(np.arange(spectrum.size, dtype=np.float64) + 1.0)
        signal = np.fft.irfft(spectrum, n=n_samples)
        peak = np.max(np.abs(signal))
        if peak > 1e-12:
            signal = 0.35 * signal / peak
        return signal.astype(np.float64), int(sample_rate)

    signal = rng.normal(0.0, 0.25, n_samples)
    return signal.astype(np.float64), int(sample_rate)


def decode_wav_array(audio) -> np.ndarray:
    raw_audio = np.asarray(audio)
    if np.issubdtype(raw_audio.dtype, np.unsignedinteger):
        info = np.iinfo(raw_audio.dtype)
        midpoint = 0.5 * (info.max + info.min)
        scale = 0.5 * (info.max - info.min)
        audio_float = (raw_audio.astype(np.float64) - midpoint) / scale
    elif np.issubdtype(raw_audio.dtype, np.integer):
        info = np.iinfo(raw_audio.dtype)
        audio_float = raw_audio.astype(np.float64) / max(abs(info.min), info.max)
    else:
        audio_float = raw_audio.astype(np.float64)
    audio = ensure_mono(audio_float)
    return ensure_even_length(audio)


def decode_uploaded_wav(uploaded_file) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(io.BytesIO(uploaded_file.getvalue()))
    return decode_wav_array(audio), int(sample_rate)


def load_wav_file(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(path)
    return decode_wav_array(audio), int(sample_rate)


def trim_audio(audio_signal: np.ndarray, sample_rate: int, duration_seconds: float) -> np.ndarray:
    n_samples = min(
        np.asarray(audio_signal).size,
        max(2, int(round(float(duration_seconds) * int(sample_rate)))),
    )
    return ensure_even_length(np.asarray(audio_signal, dtype=np.float64)[:n_samples])


def normalize_audio_group(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    peak = max((float(np.max(np.abs(signal))) for signal in signals.values() if signal.size), default=0.0)
    if peak < 1e-12:
        return {label: np.zeros_like(signal, dtype=np.float32) for label, signal in signals.items()}
    return {
        label: np.asarray(np.clip(signal / peak, -1.0, 1.0), dtype=np.float32)
        for label, signal in signals.items()
    }


def fem_stl_to_transmission_amplitude(stl_db: np.ndarray) -> np.ndarray:
    """Convert a diffuse STL curve [dB] to a pressure transmission amplitude.

    STL = 10*log10(1/tau_power) with tau_power the power transmission
    coefficient, so the amplitude transmission is sqrt(tau_power) = 10^(-STL/20).
    """
    return np.power(10.0, -np.asarray(stl_db, dtype=np.float64) / 20.0)


def auralize_window_filters(
    audio_signal: np.ndarray,
    input_rate: int,
    infinite,
    finite_results: dict[str, object],
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
):
    fem_lrm_curves = fem_lrm_curves or {}
    max_filter_frequency = float(infinite.freqs_hz[-1])
    for fem_freqs, _ in fem_lrm_curves.values():
        if fem_freqs.size:
            max_filter_frequency = max(max_filter_frequency, float(np.max(fem_freqs)))
    processing_rate = choose_processing_sample_rate(max_filter_frequency, int(input_rate))
    infinite_audio = auralize_with_frf(
        audio_signal=audio_signal,
        input_rate=int(input_rate),
        frf_freqs=infinite.freqs_hz,
        frf_response=infinite.pressure_frf,
        response_kind="amplitude",
        processing_rate=processing_rate,
        auto_resample=False,
    )
    finite_audio = {
        paper_name: auralize_with_frf(
            audio_signal=audio_signal,
            input_rate=int(input_rate),
            frf_freqs=result.freqs_hz,
            frf_response=result.pressure_frf,
            response_kind="amplitude",
            processing_rate=processing_rate,
            auto_resample=False,
        )
        for paper_name, result in finite_results.items()
    }
    fem_audio = {
        paper_name: auralize_with_frf(
            audio_signal=audio_signal,
            input_rate=int(input_rate),
            frf_freqs=fem_freqs,
            frf_response=fem_stl_to_transmission_amplitude(fem_stl),
            response_kind="amplitude",
            processing_rate=processing_rate,
            auto_resample=False,
        )
        for paper_name, (fem_freqs, fem_stl) in fem_lrm_curves.items()
        if fem_freqs.size
    }
    return infinite_audio, finite_audio, fem_audio


def build_filter_signature(
    values: dict[str, object],
    infinite,
    finite_results: dict[str, object],
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple:
    numeric_keys = (
        "f_min_hz",
        "f_max_hz",
        "n_points",
        "resonance_hz",
        "mass_ratio",
        "resonator_loss_factor",
        "unit_cell_a_m",
        "unit_cell_b_m",
        "density",
        "thickness_m",
        "young_modulus",
        "poisson_ratio",
        "host_loss_factor",
        "theta_oblique_deg",
        "theta_limit_deg",
        "theta_samples",
        "radial_samples",
    )
    return (
        values["selected_windows"],
        values["incidence"],
        tuple(round(float(values[key]), 8) for key in numeric_keys),
        (
            values.get("material"),
            values.get("partition_type", "single"),
            round(float(values.get("cavity_thickness_m", 0.0) or 0.0), 8),
            values.get("second_material"),
            (
                None
                if values.get("second_thickness_m") is None
                else round(float(values["second_thickness_m"]), 8)
            ),
            values.get("second_plate_type"),
        ),
        round(float(np.mean(infinite.stl_db)), 8),
        tuple(
            (paper_name, round(float(np.mean(result.stl_db)), 8))
            for paper_name, result in finite_results.items()
        ),
        tuple(
            (paper_name, round(float(np.mean(stl)), 8))
            for paper_name, (_, stl) in sorted((fem_lrm_curves or {}).items())
        ),
    )


def render_sidebar(cache_points: list[dict[str, object]]):
    with st.sidebar:
        st.header("Window")
        selected_windows = st.multiselect(
            "Paper size",
            options=list(DEFAULT_PAPER_WINDOW_NAMES),
            default=list(DEFAULT_PAPER_WINDOW_NAMES),
        )
        if not selected_windows:
            selected_windows = ["A4"]
        show_fem_lrm = st.checkbox(
            "Overlay legacy FEM CSV curves",
            value=False,
            help=(
                "Overlay the old Simcenter3D/OpenCFS CSV exports for the selected "
                "paper windows. These are independent of the cache point controls."
            ),
        )

        st.header("Resonator")
        cached_frequencies = cached_resonance_frequencies(cache_points)
        if cached_frequencies:
            default_index = 0
            if DEFAULT_RESONANCE_HZ in cached_frequencies:
                default_index = cached_frequencies.index(DEFAULT_RESONANCE_HZ)
            resonance_hz = st.select_slider(
                "Cached resonance frequency [Hz]",
                options=cached_frequencies,
                value=cached_frequencies[default_index],
                format_func=lambda value: f"{float(value):.0f} Hz",
            )
        else:
            resonance_hz = st.slider(
                "Resonance frequency [Hz]", 100.0, 2000.0, DEFAULT_RESONANCE_HZ, 10.0
            )

        st.caption(
            "Fixed scan: mass ratio 20%, structural loss 0.05, "
            "50 x 50 mm unit cell, single aluminum partition."
        )

        with st.expander("Frequency And Integration", expanded=False):
            f_min_hz = st.number_input("Minimum frequency [Hz]", min_value=1.0, value=10.0, step=10.0)
            f_max_hz = st.number_input("Maximum frequency [Hz]", min_value=20.0, value=4000.0, step=100.0)
            n_points = st.slider("Frequency points", 80, 800, 260, 20)
            incidence = st.radio(
                "Incidence",
                options=("diffuse", "oblique", "normal"),
                horizontal=True,
                help=(
                    "'normal' is a single plane wave at 0 deg (a normal-incidence "
                    "check); 'oblique' is a single plane wave at the angle below; "
                    "'diffuse' averages over many plane waves up to the limit angle."
                ),
            )
            theta_oblique_deg = st.slider(
                "Oblique angle [deg]",
                0.0,
                80.0,
                0.0,
                1.0,
                disabled=incidence != "oblique",
            )
            theta_limit_deg = st.slider("Diffuse limit angle [deg]", 10.0, 90.0, 90.0, 1.0)
            theta_samples = st.slider("Angular samples", 21, 181, 81, 10)
            radial_samples = st.slider("Radial samples", 20, 600, 140, 20)

    return {
        "selected_windows": tuple(selected_windows),
        "show_fem_cache": True,
        "show_fem_lrm": bool(show_fem_lrm),
        "resonance_hz": float(resonance_hz),
        "mass_ratio": FIXED_MASS_RATIO,
        "resonator_loss_factor": FIXED_RESONATOR_LOSS_FACTOR,
        "unit_cell_a_m": FIXED_UNIT_CELL_A_M,
        "unit_cell_b_m": FIXED_UNIT_CELL_B_M,
        "material": None,
        "density": FIXED_DENSITY,
        "thickness_m": FIXED_THICKNESS_M,
        "young_modulus": FIXED_YOUNG_MODULUS,
        "poisson_ratio": FIXED_POISSON_RATIO,
        "host_loss_factor": FIXED_HOST_LOSS_FACTOR,
        "partition_type": FIXED_PARTITION_TYPE,
        "cavity_thickness_m": 0.05,
        "second_material": None,
        "second_thickness_m": None,
        "second_plate_type": None,
        "f_min_hz": f_min_hz,
        "f_max_hz": f_max_hz,
        "n_points": n_points,
        "incidence": incidence,
        "theta_oblique_deg": theta_oblique_deg,
        "theta_limit_deg": theta_limit_deg,
        "theta_samples": theta_samples,
        "radial_samples": radial_samples,
    }


def render_auralization(
    infinite,
    finite_results: dict[str, object],
    filter_signature: tuple,
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    st.subheader("Auralization")
    fem_lrm_curves = fem_lrm_curves or {}
    stored_result = st.session_state.get("window_auralization")
    if stored_result is not None and stored_result.get("filter_signature") != filter_signature:
        st.session_state.pop("window_auralization", None)

    recorded_audio = discover_recorded_audio_files()
    audio_options = [UPLOAD_AUDIO_LABEL, *recorded_audio]
    source_type = st.selectbox(
        "Audio source",
        options=audio_options,
    )
    include_fem = False
    if fem_lrm_curves:
        include_fem = st.checkbox(
            "Include FEM curves in auralization",
            value=True,
            help=(
                "Auralize the finite-element curves by converting their "
                "diffuse STL [dB] to a transmission amplitude (10^(-STL/20)) and "
                "running it through the same minimum-phase filter pipeline."
            ),
        )
    duration_seconds = st.slider("Audio duration [s]", 5.0, 10.0, 8.0, 0.5)
    uploaded_file = None
    if source_type == UPLOAD_AUDIO_LABEL:
        uploaded_file = st.file_uploader("WAV file", type=["wav"])

    if st.button("Run Auralization", type="primary"):
        if source_type == UPLOAD_AUDIO_LABEL:
            if uploaded_file is None:
                st.warning("Upload a WAV file first.")
                return
            audio_signal, sample_rate = decode_uploaded_wav(uploaded_file)
            audio_signal = trim_audio(audio_signal, sample_rate, duration_seconds)
        else:
            audio_signal, sample_rate = load_wav_file(recorded_audio[source_type])
            audio_signal = trim_audio(audio_signal, sample_rate, duration_seconds)

        with st.spinner("Building windowed audio..."):
            fem_curves_for_audio = fem_lrm_curves if include_fem else {}
            infinite_audio, finite_audio, fem_audio = auralize_window_filters(
                audio_signal,
                sample_rate,
                infinite,
                finite_results,
                fem_curves_for_audio,
            )
            playback_signals = {
                "Original": infinite_audio.input_signal,
                "Infinite panel": infinite_audio.output_signal,
            }
            playback_signals.update(
                {
                    paper_window_label(paper_name): result.output_signal
                    for paper_name, result in finite_audio.items()
                }
            )
            playback_signals.update(
                {
                    fem_curve_display_label(paper_name): result.output_signal
                    for paper_name, result in fem_audio.items()
                }
            )
            st.session_state["window_auralization"] = {
                "signals": normalize_audio_group(playback_signals),
                "sample_rate": infinite_audio.sample_rate,
                "debug": infinite_audio.sample_rate_debug,
                "filter_signature": filter_signature,
            }

    result_state = st.session_state.get("window_auralization")
    if result_state is None:
        return

    debug = result_state["debug"]
    st.caption(
        "Internal sample rate: "
        f"{debug.processing_rate} Hz "
        f"({debug.selection_mode}, input {debug.input_rate} Hz, max FRF {debug.max_frequency:.1f} Hz)."
    )
    for label, signal in result_state["signals"].items():
        st.caption(label)
        st.audio(signal, sample_rate=result_state["sample_rate"], format="audio/wav")


def _format_cache_point(point: dict[str, object]) -> str:
    return (
        f"f_res={float(point['f_res_hz']):.1f} Hz, "
        f"m_ratio={100.0 * float(point['m_ratio']):.1f}%, "
        f"eta={float(point['eta_res']):.3f}"
    )


def render_fem_cache_panel(
    values: dict[str, object],
    fem_lookup: dict[str, object] | None,
    cache_points: list[dict[str, object]],
    cache_curve: tuple[np.ndarray, np.ndarray] | None,
    cache_error: str | None,
) -> None:
    st.subheader("FEM Cache")
    if cache_error:
        st.caption(cache_error)
    if fem_lookup is None:
        st.info("No fem-prerun.json/base DAT was found, so FEM cache lookup is disabled.")
        return
    if not cache_points:
        st.info("No FEM cache points are in the shared store yet.")
        return

    exact_point = next(
        (
            point
            for point in cache_points
            if fem_cache_point_matches(
                point,
                float(values["resonance_hz"]),
                float(values["mass_ratio"]),
                float(values["resonator_loss_factor"]),
            )
        ),
        None,
    )
    if cache_curve is not None:
        point = exact_point
        label = _format_cache_point(point) if point is not None else "current controls"
        st.success(f"Loaded cached FEM curve: {label}.")
    elif exact_point is not None:
        st.warning(
            "A FEM point with the current coordinates exists, but it did not match "
            "the active base-DAT cache signature."
        )
    else:
        nearest = nearest_fem_cache_point(
            cache_points,
            float(values["resonance_hz"]),
            float(values["mass_ratio"]),
            float(values["resonator_loss_factor"]),
        )
        if nearest is not None:
            st.info(f"Nearest buffered point: {_format_cache_point(nearest)}.")

    table = build_fem_cache_points_table(cache_points)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "f_res [Hz]": st.column_config.NumberColumn(format="%.1f"),
            "m_ratio [%]": st.column_config.NumberColumn(format="%.1f"),
            "eta_res [-]": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="FEM Analytical Filter Window App", layout="wide")
    st.title("FEM/Analytical Resonator Filter")

    fem_lookup = prepare_fem_cache_lookup()
    cache_points, cache_error = load_fem_cache_points()
    values = render_sidebar(cache_points)
    show_fem_cache = values.pop("show_fem_cache", False)
    show_fem_lrm = values.pop("show_fem_lrm", False)
    try:
        with st.spinner("Updating analytical filter..."):
            infinite, finite_results = compute_curves(**values)
    except ValueError as exc:
        st.error(str(exc))
        return

    available_fem_csv = load_fem_lrm_curves()
    fem_lrm_curves = {
        window: available_fem_csv[window]
        for window in values["selected_windows"]
        if window in available_fem_csv
    }
    if show_fem_lrm:
        if not available_fem_csv:
            st.info(f"No FEM LRM curves found in {FEM_RESULTS_DIR}.")
        elif not fem_lrm_curves:
            st.info(
                "No legacy FEM CSV data for the selected windows. "
                f"Available: {', '.join(sorted(available_fem_csv))}."
            )

    fem_cache_curve = (
        load_current_fem_cache_curve(
            fem_lookup,
            values["resonance_hz"],
            values["mass_ratio"],
            values["resonator_loss_factor"],
        )
        if show_fem_cache
        else None
    )
    fem_overlay_curves = {}
    if show_fem_lrm:
        fem_overlay_curves.update(fem_lrm_curves)
    if show_fem_cache and fem_cache_curve is not None:
        fem_overlay_curves[FEM_CACHE_CURVE_KEY] = fem_cache_curve

    st.plotly_chart(
        build_plot(
            infinite,
            finite_results,
            values["resonance_hz"],
            fem_overlay_curves,
        ),
        width="stretch",
    )

    render_fem_cache_panel(values, fem_lookup, cache_points, fem_cache_curve, cache_error)

    render_auralization(
        infinite,
        finite_results,
        build_filter_signature(values, infinite, finite_results, fem_overlay_curves),
        fem_overlay_curves,
    )


if __name__ == "__main__":
    main()
