from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
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
DEFAULT_F_MIN_HZ = 20.0
DEFAULT_F_MAX_HZ = 5000.0
DEMO_WINDOW_NAME = "A4"
DEMO_N_POINTS = 260
DEMO_INCIDENCE = "diffuse"
ABOUT_AUTHOR = "Jiahua Zhang, PhD Candidate at KU Leuven/Siemens."
ABOUT_CONTACT = "jiahua.zhang@siemens.com"
ABOUT_ACKNOWLEDGEMENTS = (
    "The European Commission is gratefully acknowledged for their support of the "
    "Horizon Europe DN METAVISION project (GA 101072415). Views and opinions expressed "
    "are however those of the authors only and do not necessarily reflect those of the "
    "European Union. The European Union cannot be held responsible for them. The research "
    "of L. Van Belle (fellowship no. 1254325N) is funded by a grant from the Research "
    "Foundation -- Flanders (FWO)."
)
BRAND_LOGOS = (
    {"label": "KU Leuven", "relative_path": "archive/kuleuven.jpg", "width": 96},
    {"label": "Siemens", "relative_path": "archive/siemens.png", "width": 210},
    {"label": "MSCA METAVISION", "relative_path": "archive/metavision.png", "width": 220},
)
REFERENCE_ITEMS = (
    {
        "label": "Claeys, C., Deckers, E., Pluymers, B., & Desmet, W. (2016). "
        "A lightweight vibro-acoustic metamaterial demonstrator: Numerical and experimental investigation. "
        "Mechanical Systems and Signal Processing, 70-71, 853-880.",
        "url": "https://doi.org/10.1016/j.ymssp.2015.08.029",
    },
    {
        "label": "Van Belle, L., Claeys, C., Deckers, E., & Desmet, W. (2019). "
        "The impact of damping on the sound transmission loss of locally resonant metamaterial plates. "
        "Journal of Sound and Vibration, 461, 114909.",
        "url": "https://doi.org/10.1016/j.jsv.2019.114909",
    },
    {
        "label": "Fredianelli, L., Artuso, F., Pompei, G., Licitra, G., Iannace, G., & Akbaba, A. (2025). "
        "DataSEC - Dataset for Sound Event Classification of environmental noise. Zenodo.",
        "url": "https://doi.org/10.5281/zenodo.17033970",
    },
)

# openCFS FEM LRM (local-resonant metamaterial) STL exports, diffuse incidence.
FEM_RESULTS_DIR = PROJECT_ROOT / "fem" / "results"
_FEM_LRM_FILENAME_RE = re.compile(r"^(A\d)_wall_lrm\.csv$", re.IGNORECASE)
FEM_CACHE_CURVE_KEY = "Cache"
UPLOAD_AUDIO_LABEL = "Upload WAV"
RECORDED_AUDIO_CANDIDATES = (
    ("DATASEC hairdryer 0004", "archive/Recordings/DATASEC/hairdryer-0004.wav"),
    ("DATASEC motorbike idling 0001", "archive/Recordings/DATASEC/motorbike idling-0001.wav"),
    ("DATASEC vacuum cleaner 0001", "archive/Recordings/DATASEC/vacuum cleaner-0001.wav"),
    ("SkyExpress Flight (Try to tune at 420 Hz)", "archive/Recordings/SkyExpress_Flight.wav")
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
    if result is None:
        fallback = find_matching_fem_entry(
            store,
            f_res_hz=resonance_hz,
            m_ratio=mass_ratio,
            eta_res=resonator_loss_factor,
            base_dat_hash=None,
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
        return "FEM reference"
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


def get_resource_path(relative_name: str) -> Path:
    """Resolve bundled project resources from source or packaged builds."""
    candidate = PROJECT_ROOT / relative_name
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[2] / relative_name


def get_brand_logo_specs() -> list[dict[str, object]]:
    return [
        {
            "label": logo["label"],
            "path": get_resource_path(str(logo["relative_path"])),
            "width": int(logo["width"]),
        }
        for logo in BRAND_LOGOS
    ]


def build_demo_values(resonance_hz: float) -> dict[str, object]:
    return {
        "selected_windows": (DEMO_WINDOW_NAME,),
        "show_fem_cache": True,
        "show_fem_lrm": False,
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
        "f_min_hz": DEFAULT_F_MIN_HZ,
        "f_max_hz": DEFAULT_F_MAX_HZ,
        "n_points": DEMO_N_POINTS,
        "incidence": DEMO_INCIDENCE,
        "theta_oblique_deg": 0.0,
        "theta_limit_deg": 90.0,
        "theta_samples": 81,
        "radial_samples": 140,
    }


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
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
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
    f_min = float(np.min(freq_values)) if f_min_hz is None else float(f_min_hz)
    f_max = float(np.max(freq_values)) if f_max_hz is None else float(f_max_hz)
    if f_min <= 0.0:
        raise ValueError("Plot minimum frequency must be positive.")
    if f_max <= f_min:
        raise ValueError("Plot maximum frequency must be greater than minimum frequency.")
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


def audio_level_db(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def build_waveform_figure(signal: np.ndarray, sample_rate: int, color: str) -> go.Figure:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        x = np.array([0.0])
        y = np.array([0.0])
    else:
        step = max(1, int(np.ceil(signal.size / 900)))
        y = signal[::step]
        x = np.arange(y.size, dtype=np.float64) * step / float(sample_rate)
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": color, "width": 1.4},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=112,
        margin={"l": 4, "r": 4, "t": 2, "b": 2},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False, "range": [-1.02, 1.02]},
    )
    return fig


def build_audio_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    *,
    max_frequency_hz: float = DEFAULT_F_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 2:
        return np.array([0.0]), np.array([0.0]), np.array([[-120.0]])

    frame_size = min(2048, max(256, int(2 ** np.floor(np.log2(signal.size)))))
    hop_size = max(1, frame_size // 4)
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size))
    starts = np.arange(0, signal.size - frame_size + 1, hop_size, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    window = np.hanning(frame_size)
    frames = np.stack([signal[start : start + frame_size] * window for start in starts], axis=0)
    spectrum = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    keep = freqs <= min(float(max_frequency_hz), float(sample_rate) / 2.0)
    freqs = freqs[keep]
    magnitude = np.abs(spectrum[:, keep]).T
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-9))
    peak = float(np.nanmax(magnitude_db))
    spectrogram_db = np.clip(magnitude_db - peak, -80.0, 0.0)
    times = (starts.astype(np.float64) + 0.5 * frame_size) / float(sample_rate)
    return times, freqs, spectrogram_db


def build_audio_card_figure(signal: np.ndarray, sample_rate: int, color: str) -> go.Figure:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        time_x = np.array([0.0])
        time_y = np.array([0.0])
    else:
        step = max(1, int(np.ceil(signal.size / 900)))
        time_y = signal[::step]
        time_x = np.arange(time_y.size, dtype=np.float64) * step / float(sample_rate)
    spectrogram_t, spectrogram_f, spectrogram_db = build_audio_spectrogram(signal, sample_rate)
    max_spectrogram_hz = min(DEFAULT_F_MAX_HZ, float(sample_rate) / 2.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=time_y,
            mode="lines",
            line={"color": "#7dd3fc", "width": 1.1},
            hovertemplate="t=%{x:.2f}s<br>amp=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=spectrogram_t,
            y=spectrogram_f,
            z=spectrogram_db,
            zmin=-80.0,
            zmax=0.0,
            colorscale=[
                [0.0, "#020617"],
                [0.18, "#111827"],
                [0.35, "#312e81"],
                [0.52, "#7e22ce"],
                [0.68, "#db2777"],
                [0.84, "#f97316"],
                [1.0, "#fff7ad"],
            ],
            showscale=False,
            hovertemplate="t=%{x:.2f}s<br>f=%{y:.0f}Hz<br>%{z:.1f}dB<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=220,
        margin={"l": 10, "r": 10, "t": 6, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#101828",
        showlegend=False,
        font={"color": "#cbd5e1", "size": 10},
    )
    fig.update_xaxes(
        title_text="Time [s]",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(visible=False, range=[-1.02, 1.02], row=1, col=1)
    fig.update_xaxes(
        title_text="Time [s]",
        title_standoff=4,
        range=[float(time_x[0]), float(time_x[-1]) if time_x.size else 0.0],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Frequency [Hz]",
        range=[0.0, max_spectrogram_hz],
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    return fig


def build_audio_output_label(paper_name: str) -> str:
    if paper_name == DEMO_WINDOW_NAME:
        return "A4 size analytical"
    return f"{paper_window_label(paper_name)} analytical"


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


def render_demo_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .demo-top-rule {
            border-top: 1px solid #d9dde5;
            margin: 1.0rem 0 1.2rem;
        }
        .demo-kicker {
            color: #5f6878;
            font-size: 0.86rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .demo-meta {
            color: #4f5665;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .audio-card-title {
            font-weight: 700;
            color: #202635;
            margin-bottom: 0.15rem;
        }
        .audio-card-caption {
            color: #667085;
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    title_col, logos_col = st.columns([1.65, 1.35], gap="large")
    with title_col:
        st.title("Auralization Comparison: Local Resonant Metamaterial Analytical vs FEM Response")
    with logos_col:
        st.caption("Project Partners")
        logo_columns = st.columns(len(BRAND_LOGOS), gap="medium")
        for column, spec in zip(logo_columns, get_brand_logo_specs()):
            with column:
                path = Path(spec["path"])
                if path.exists():
                    st.image(str(path), width=int(spec["width"]))
                else:
                    st.caption(str(spec["label"]))
    st.markdown('<div class="demo-top-rule"></div>', unsafe_allow_html=True)


def render_demo_controls(cache_points: list[dict[str, object]]) -> dict[str, object]:
    control_col, info_col = st.columns([1.0, 1.45], gap="large")
    with control_col:
        st.markdown('<div class="demo-kicker">Demonstration setting</div>', unsafe_allow_html=True)
        cached_frequencies = cached_resonance_frequencies(cache_points)
        if cached_frequencies:
            default_index = 0
            if DEFAULT_RESONANCE_HZ in cached_frequencies:
                default_index = cached_frequencies.index(DEFAULT_RESONANCE_HZ)
            resonance_hz = st.select_slider(
                "Resonance frequency [Hz]",
                options=cached_frequencies,
                value=cached_frequencies[default_index],
                format_func=lambda value: f"{float(value):.0f} Hz",
            )
        else:
            resonance_hz = st.slider(
                "Resonance frequency [Hz]", 100.0, 2000.0, DEFAULT_RESONANCE_HZ, 10.0
            )
    with info_col:
        st.markdown(
            (
                '<div class="demo-meta">'
                "<strong>A4 sized panel</strong><br>"
                "Analytical infinite and finite-window local resonant metamaterial responses are compared "
                "against the matching diffuse-field Finite Element Method (FEM) reference calculated with Siemens Simcenter3D. "
                f"The display range is {DEFAULT_F_MIN_HZ:.0f} Hz to {DEFAULT_F_MAX_HZ / 1000.0:.1f} kHz."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    return build_demo_values(float(resonance_hz))


def render_auralization(
    infinite,
    finite_results: dict[str, object],
    filter_signature: tuple,
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    st.divider()
    st.header("Auralization comparison")
    fem_lrm_curves = fem_lrm_curves or {}
    stored_result = st.session_state.get("window_auralization")
    if stored_result is not None and stored_result.get("filter_signature") != filter_signature:
        st.session_state.pop("window_auralization", None)

    recorded_audio = discover_recorded_audio_files()
    audio_options = [UPLOAD_AUDIO_LABEL, *recorded_audio]
    source_col, duration_col, action_col = st.columns([1.45, 1.0, 0.85], gap="large")
    with source_col:
        source_type = st.selectbox("Audio source", options=audio_options)
    with duration_col:
        duration_seconds = st.slider("Duration [s]", 5.0, 10.0, 8.0, 0.5)
    uploaded_file = None
    if source_type == UPLOAD_AUDIO_LABEL:
        uploaded_file = st.file_uploader("WAV file", type=["wav"])

    with action_col:
        st.write("")
        run_auralization = st.button("Run comparison", type="primary", width="stretch")

    if run_auralization:
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
            infinite_audio, finite_audio, fem_audio = auralize_window_filters(
                audio_signal,
                sample_rate,
                infinite,
                finite_results,
                fem_lrm_curves,
            )
            playback_signals = {
                "Original": infinite_audio.input_signal,
                "Infinite panel": infinite_audio.output_signal,
            }
            playback_signals.update(
                {
                    build_audio_output_label(paper_name): result.output_signal
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
        f"Processed at {debug.processing_rate} Hz "
        f"({debug.selection_mode}; input {debug.input_rate} Hz; response to {debug.max_frequency:.1f} Hz)."
    )
    signals = result_state["signals"]
    sample_rate = int(result_state["sample_rate"])
    original_level = audio_level_db(next(iter(signals.values())))
    colors = ["#475467", "#1f77b4", "#8c564b", "#111827", "#2ca02c", "#9467bd"]
    columns = st.columns(2, gap="large")
    for index, (label, signal) in enumerate(signals.items()):
        with columns[index % 2]:
            with st.container(border=True):
                level = audio_level_db(signal)
                delta = level - original_level
                st.markdown(f'<div class="audio-card-title">{label}</div>', unsafe_allow_html=True)
                st.markdown(
                    (
                        '<div class="audio-card-caption">'
                        f"RMS level {level:.1f} dBFS, {delta:+.1f} dB vs original"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    build_audio_card_figure(signal, sample_rate, colors[index % len(colors)]),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.audio(signal, sample_rate=sample_rate, format="audio/wav")


def render_project_info_section() -> None:
    st.divider()
    with st.expander("Project information", expanded=False):
        st.markdown(
            f"""
            **Author**
            {ABOUT_AUTHOR}

            **Contact**
            {ABOUT_CONTACT}

            **Acknowledgements**
            {ABOUT_ACKNOWLEDGEMENTS}
            """
        )
    with st.expander("References", expanded=False):
        st.markdown(
            "\n".join(
                f"- [{item['label']}]({item['url']})"
                for item in REFERENCE_ITEMS
            )
        )


def main() -> None:
    st.set_page_config(page_title="Auralization Comparison", layout="wide")
    render_demo_styles()
    render_brand_header()

    fem_lookup = prepare_fem_cache_lookup()
    cache_points, cache_error = load_fem_cache_points()
    del cache_error
    values = render_demo_controls(cache_points)
    show_fem_cache = values.pop("show_fem_cache", False)
    values.pop("show_fem_lrm", False)
    try:
        with st.spinner("Updating analytical filter..."):
            infinite, finite_results = compute_curves(**values)
    except ValueError as exc:
        st.error(str(exc))
        return

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
    if show_fem_cache and fem_cache_curve is not None:
        fem_overlay_curves[FEM_CACHE_CURVE_KEY] = fem_cache_curve

    st.plotly_chart(
        build_plot(
            infinite,
            finite_results,
            values["resonance_hz"],
            fem_overlay_curves,
            f_min_hz=values["f_min_hz"],
            f_max_hz=values["f_max_hz"],
        ),
        width="stretch",
    )

    render_auralization(
        infinite,
        finite_results,
        build_filter_signature(values, infinite, finite_results, fem_overlay_curves),
        fem_overlay_curves,
    )
    render_project_info_section()


if __name__ == "__main__":
    main()
