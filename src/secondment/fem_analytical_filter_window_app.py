from __future__ import annotations

import base64
import io
import mimetypes
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
from secondment.audio_compare_component import render_audio_comparison
from secondment.compare_analytical_filter_window import (
    PAPER_WINDOW_SIZES_M,
    paper_window_area_m2,
    paper_window_label,
)
from secondment.materials import get_material
from secondment.numerical_store import (
    get_numerical_store_path,
    load_numerical_result,
)
from secondment.psychoacoustic_metrics import (
    NativePsychoacousticError,
    TrackPsychoacoustics,
    compute_psychoacoustic_comparison,
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
COLOR_ANALYTICAL = "#007C83"
COLOR_FEM = "#B84E32"
COLOR_INFINITE = "#7A8793"
COLOR_INK = "#102A43"
COLOR_MUTED = "#52616F"
COLOR_GRID = "#D8E2EA"
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
DEFAULT_RESONANCE_HZ = 420.0
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
        "label": "Van Belle, L., Di Giusto, F., & Deckers, E. (2026). "
        "Model-based auralization of vibro-acoustic metamaterial partitions. "
        "In Proceedings of Forum Acusticum - Euronoise 2025 (pp. 5399-5406).",
        "url": "https://doi.org/10.61782/fa.2025.0574",
    },
    {
        "label": "Van Belle, L., Claeys, C., Deckers, E., & Desmet, W. (2019). "
        "The impact of damping on the sound transmission loss of locally resonant metamaterial plates. "
        "Journal of Sound and Vibration, 461, 114909.",
        "url": "https://doi.org/10.1016/j.jsv.2019.114909",
    },
    {
        "label": "Zhang, J., Cuenca, J., De Ryck, L., Van Belle, L., & Deckers, E. (2026). "
        "Psychoacoustical design optimization for acoustic metamaterials using global sensitivity analysis. "
        "In Proceedings of Forum Acusticum (pp. 2369-2373).",
        "url": "https://doi.org/10.61782/fa.2025.0715",
    },
    {
        "label": "Fredianelli, L., Artuso, F., Pompei, G., Licitra, G., Iannace, G., & Akbaba, A. (2025). "
        "DataSEC - Dataset for Sound Event Classification of environmental noise. Zenodo.",
        "url": "https://doi.org/10.5281/zenodo.17033970",
    },
    {
        "label": "DIN 45692:2009-08. Measurement technique for the simulation of the auditory sensation of sharpness.",
        "url": "https://www.dinmedia.de/en/standard/din-45692/117635111",
    },
    {
        "label": "ISO/TS 20065:2022. Acoustics: Objective method for assessing the audibility of tones in noise, engineering method.",
        "url": "https://www.iso.org/standard/81518.html",
    },
    {
        "label": "Aures, W. (1985). Berechnungsverfahren für den sensorischen Wohlklang beliebiger Schallsignale. Acustica, 59, 130-141.",
        "url": "https://terhardt.userweb.mwn.de/ter/ref/Aures1985b.html",
    },
)

# openCFS FEM LRM (local-resonant metamaterial) STL exports, diffuse incidence.
FEM_RESULTS_DIR = PROJECT_ROOT / "fem" / "results"
_FEM_LRM_FILENAME_RE = re.compile(r"^(A\d)_wall_lrm\.csv$", re.IGNORECASE)
FEM_CACHE_CURVE_KEY = "Cache"
UPLOAD_AUDIO_LABEL = "Use your own WAV"
RECORDED_AUDIO_CANDIDATES = (
    ("Aircraft cabin (works well at 420 Hz)", "archive/Recordings/SkyExpress_Flight.wav"),
    ("Hair dryer (works well at 850 Hz)", "archive/Recordings/DATASEC/hairdryer-0004.wav"),
    ("Grinder (works well at 560 Hz)", "archive/Recordings/DATASEC/grinder-0008.wav"),
    ("Vacuum cleaner (works well at 770 Hz)", "archive/Recordings/DATASEC/vacuum cleaner-0001.wav"),
)
METAVISION_DEMOS_URL = "https://www.heu-metavision.eu/dissemination/demos/"
ORIGINAL_AUDIO_LABEL = "Original"
BARE_PANEL_AUDIO_LABEL = "Bare panel: analytical"
ANALYTICAL_PANEL_AUDIO_LABEL = "Metamaterial: analytical"
FEM_PANEL_AUDIO_LABEL = "Metamaterial: FEM simulation"
INFINITE_PANEL_AUDIO_LABEL = "Ideal infinite panel: analytical simulation"
PRIMARY_AUDIO_LABELS = (
    ORIGINAL_AUDIO_LABEL,
    BARE_PANEL_AUDIO_LABEL,
    ANALYTICAL_PANEL_AUDIO_LABEL,
    FEM_PANEL_AUDIO_LABEL,
)
DIAGNOSTIC_AUDIO_LABELS = (
    ORIGINAL_AUDIO_LABEL,
    INFINITE_PANEL_AUDIO_LABEL,
    ANALYTICAL_PANEL_AUDIO_LABEL,
    FEM_PANEL_AUDIO_LABEL,
)
PSYCHOACOUSTIC_DISPLAY_LABELS = {
    ORIGINAL_AUDIO_LABEL: "Original",
    BARE_PANEL_AUDIO_LABEL: "Bare panel<br>(analytical)",
    ANALYTICAL_PANEL_AUDIO_LABEL: "Metamaterial<br>(analytical)",
    FEM_PANEL_AUDIO_LABEL: "Metamaterial<br>(FEM)",
}


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
        return FEM_PANEL_AUDIO_LABEL
    return f"FEM simulation: {key} panel"


def fem_curve_color(key: str) -> str:
    if key == FEM_CACHE_CURVE_KEY:
        return COLOR_FEM
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


@st.cache_data(show_spinner=False)
def resource_data_uri(relative_name: str) -> str | None:
    """Return a bundled image as a data URI for accessible app-owned markup."""
    path = get_resource_path(relative_name)
    if not path.exists():
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
        bare_leaves = (
            PartitionLeaf(host=host, resonator=None),
            PartitionLeaf(host=second_host, resonator=None),
        )
        cavities = (AirCavity(thickness_m=cavity_thickness_m),)

        def filter_call(size_type: str, **window_kwargs):
            return partition_pressure_filter(
                freqs, leaves, cavities, size_type=size_type, **window_kwargs, **common_kwargs
            )

        def bare_filter_call(size_type: str, **window_kwargs):
            return partition_pressure_filter(
                freqs,
                bare_leaves,
                cavities,
                size_type=size_type,
                **window_kwargs,
                **common_kwargs,
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

        def bare_filter_call(size_type: str, **window_kwargs):
            return bending_panel_pressure_filter(
                freqs,
                host=host,
                resonator=None,
                size_type=size_type,
                **window_kwargs,
                **common_kwargs,
            )

    infinite = filter_call("infinite")
    finite = {
        paper_name: filter_call("finite", window_area_m2=paper_window_area_m2(paper_name))
        for paper_name in selected_windows
    }
    bare_finite = {
        paper_name: bare_filter_call(
            "finite",
            window_area_m2=paper_window_area_m2(paper_name),
        )
        for paper_name in selected_windows
    }
    return infinite, finite, bare_finite


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
            name="Ideal infinite panel: analytical",
            mode="lines",
            line={"color": COLOR_INFINITE, "width": 2.5, "dash": "dash"},
            hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB<extra>Infinite analytical</extra>",
        )
    )
    for paper_name, result in finite_results.items():
        freq_arrays.append(np.asarray(result.freqs_hz, dtype=np.float64))
        stl_arrays.append(np.asarray(result.stl_db, dtype=np.float64))
        fig.add_trace(
            go.Scatter(
                x=result.freqs_hz,
                y=result.stl_db,
                name=f"{paper_name} metamaterial: analytical",
                mode="lines",
                line={"color": COLOR_ANALYTICAL, "width": 3},
                hovertemplate=(
                    "%{x:.0f} Hz<br>%{y:.1f} dB"
                    f"<extra>{paper_name} metamaterial analytical</extra>"
                ),
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
                hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB<extra>FEM simulation</extra>",
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
            name="Selected target frequency",
            mode="lines",
            line={"color": COLOR_INK, "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[resonance_hz],
            y=[y1],
            text=[f"Target {resonance_hz:.0f} Hz"],
            mode="text",
            textposition="top center",
            textfont={"color": COLOR_INK, "size": 12},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=430,
        margin={"l": 56, "r": 24, "t": 40, "b": 88},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_MUTED, "size": 12},
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [float(np.log10(f_min)), float(np.log10(f_max))],
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "minor": {"showgrid": True, "gridcolor": "rgba(216,226,234,0.5)"},
            "zeroline": False,
        },
        yaxis={
            "title": "Transmission loss (dB)",
            "range": [y0, y1],
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "zeroline": False,
        },
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "font": {"size": 11}},
        hovermode="x unified",
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
        return np.array([0.0]), np.array([1.0]), np.array([[-120.0]])

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
    frequency_ceiling = min(float(max_frequency_hz), float(sample_rate) / 2.0)
    keep = (freqs > 0.0) & (freqs <= frequency_ceiling)
    if not np.any(keep):
        times = (starts.astype(np.float64) + 0.5 * frame_size) / float(sample_rate)
        return (
            times,
            np.array([max(1.0, frequency_ceiling)]),
            np.full((1, times.size), -80.0),
        )
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
        type="log",
        range=[
            float(np.log10(max(1.0, float(spectrogram_f[0])))),
            float(np.log10(max(1.0, max_spectrogram_hz))),
        ],
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    return fig


def build_audio_output_label(paper_name: str) -> str:
    if paper_name == DEMO_WINDOW_NAME:
        return ANALYTICAL_PANEL_AUDIO_LABEL
    return f"{paper_window_label(paper_name)}: analytical simulation"


def build_bare_audio_output_label(paper_name: str) -> str:
    if paper_name == DEMO_WINDOW_NAME:
        return BARE_PANEL_AUDIO_LABEL
    return f"Bare {paper_window_label(paper_name)}: analytical simulation"


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
    bare_results: dict[str, object],
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
    bare_audio = {
        paper_name: auralize_with_frf(
            audio_signal=audio_signal,
            input_rate=int(input_rate),
            frf_freqs=result.freqs_hz,
            frf_response=result.pressure_frf,
            response_kind="amplitude",
            processing_rate=processing_rate,
            auto_resample=False,
        )
        for paper_name, result in bare_results.items()
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
    return infinite_audio, bare_audio, finite_audio, fem_audio


def build_filter_signature(
    values: dict[str, object],
    infinite,
    finite_results: dict[str, object],
    bare_results: dict[str, object],
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
            (paper_name, round(float(np.mean(result.stl_db)), 8))
            for paper_name, result in bare_results.items()
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
        :root {
            --mv-ink: #102A43;
            --mv-muted: #52616F;
            --mv-border: #D8E2EA;
            --mv-surface: #F1F5F7;
            --mv-panel: #FCFDFD;
            --mv-analytical: #007C83;
            --mv-fem: #B84E32;
        }
        html {
            scroll-behavior: smooth;
        }
        .block-container {
            padding-top: 4.5rem;
            padding-bottom: 3.5rem;
            max-width: 1120px;
        }
        .st-key-listening_stage {
            background: var(--mv-panel);
            box-shadow: 0 18px 50px rgba(16, 42, 67, 0.06);
        }
        div.stButton > button {
            transition: transform 120ms ease, background-color 160ms ease;
        }
        div.stButton > button:active {
            transform: translateY(1px);
        }
        div.stButton > button:focus-visible {
            outline: 3px solid rgba(0, 124, 131, 0.25);
            outline-offset: 2px;
        }
        .mv-brand-rail {
            min-height: 3.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            border-bottom: 1px solid var(--mv-border);
        }
        .mv-brand-lockup {
            display: flex;
            align-items: center;
            gap: 1rem;
            min-width: 0;
        }
        .mv-brand-lockup img {
            display: block;
            width: 176px;
            max-height: 38px;
            object-fit: contain;
            object-position: left center;
        }
        .mv-brand-lockup span {
            color: var(--mv-muted);
            font-size: 0.92rem;
            white-space: nowrap;
        }
        .mv-brand-rail a {
            color: var(--mv-ink);
            font-weight: 700;
            text-decoration: none;
            white-space: nowrap;
        }
        .mv-brand-rail a:hover,
        .mv-brand-rail a:focus-visible {
            color: var(--mv-analytical);
            text-decoration: underline;
            text-underline-offset: 0.2rem;
        }
        .mv-hero {
            max-width: 720px;
            padding: 2.5rem 0 2.25rem;
        }
        .mv-hero h1 {
            color: var(--mv-ink);
            font-size: clamp(2.25rem, 5vw, 3.35rem);
            line-height: 1.06;
            letter-spacing: -0.035em;
            margin: 0 0 1.1rem;
            max-width: 740px;
        }
        .mv-hero-copy {
            color: var(--mv-muted);
            font-size: 1.12rem;
            line-height: 1.65;
            max-width: 700px;
            margin: 0;
        }
        .mv-hero-note {
            color: var(--mv-ink);
            font-size: 0.92rem;
            font-weight: 650;
            margin: 1.1rem 0 0;
        }
        .mv-stage-title {
            color: var(--mv-ink);
            font-size: 1.45rem;
            line-height: 1.25;
            margin: 0 0 0.25rem;
        }
        .mv-stage-copy,
        .mv-evidence-copy {
            color: var(--mv-muted);
            line-height: 1.65;
            margin: 0 0 1.25rem;
        }
        .mv-result-lede {
            border-left: 4px solid var(--mv-analytical);
            padding: 0.25rem 0 0.25rem 1.15rem;
            margin: 2.75rem 0 1.5rem;
        }
        .mv-result-lede strong {
            color: var(--mv-ink);
            display: block;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }
        .mv-result-lede p {
            color: var(--mv-muted);
            font-size: 1.02rem;
            line-height: 1.6;
            margin: 0;
        }
        .mv-model-facts {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            border-top: 1px solid var(--mv-border);
            border-bottom: 1px solid var(--mv-border);
            margin: 0.25rem 0 1.5rem;
        }
        .mv-model-fact {
            padding: 0.85rem 1rem;
            border-left: 1px solid var(--mv-border);
        }
        .mv-model-fact:first-child {
            border-left: 0;
        }
        .mv-model-fact span {
            display: block;
            color: var(--mv-muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-bottom: 0.2rem;
        }
        .mv-model-fact strong {
            color: var(--mv-ink);
            font-size: 1.08rem;
            font-variant-numeric: tabular-nums;
        }
        .audio-card-title {
            color: var(--mv-ink);
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 0.15rem;
        }
        .audio-card-caption {
            color: var(--mv-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }
        .mv-partner-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2rem 3rem;
            padding: 1.5rem 0 0.5rem;
        }
        .mv-partner {
            flex: 1 1 180px;
            min-width: 150px;
            text-align: center;
        }
        .mv-partner img {
            display: block;
            max-width: 210px;
            width: auto;
            height: 44px;
            object-fit: contain;
            margin: 0 auto;
        }
        .mv-footer-note {
            color: var(--mv-muted);
            font-size: 0.88rem;
            line-height: 1.6;
            margin-top: 1rem;
        }
        @media (max-width: 640px) {
            .block-container {
                padding: 4rem 1rem 2.5rem;
            }
            .mv-brand-rail {
                min-height: 3.25rem;
                gap: 0.75rem;
            }
            .mv-brand-lockup img {
                width: 138px;
                max-height: 32px;
            }
            .mv-brand-lockup span {
                display: none;
            }
            .mv-brand-rail a {
                font-size: 0.84rem;
            }
            .mv-hero {
                padding: 1.75rem 0 1.75rem;
            }
            .mv-hero h1 {
                font-size: 2rem;
                line-height: 1.1;
                letter-spacing: -0.025em;
            }
            .mv-hero-copy {
                font-size: 1rem;
                line-height: 1.55;
            }
            .mv-result-lede {
                margin-top: 2rem;
            }
            .mv-model-facts {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .mv-model-fact {
                padding: 0.75rem 0.65rem;
            }
            .mv-model-fact:nth-child(odd) {
                border-left: 0;
            }
            .mv-model-fact:nth-child(n + 3) {
                border-top: 1px solid var(--mv-border);
            }
            .mv-partner-strip {
                gap: 1.5rem;
            }
            .mv-partner {
                flex-basis: 120px;
                min-width: 110px;
            }
            .mv-partner img {
                max-width: 150px;
                height: 36px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_rail() -> None:
    logo_uri = resource_data_uri("archive/metavision.png")
    logo_markup = (
        f'<img src="{logo_uri}" alt="METAVISION: Metamaterials for Vibration and Sound Reduction">'
        if logo_uri
        else "<strong>METAVISION</strong>"
    )
    st.markdown(
        f"""
        <nav class="mv-brand-rail" aria-label="Demo navigation">
            <div class="mv-brand-lockup">
                {logo_markup}
                <span>Interactive listening demo</span>
            </div>
            <a href="{METAVISION_DEMOS_URL}">Back to METAVISION</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="mv-hero" aria-labelledby="mv-page-title">
            <h1 id="mv-page-title">Hear a Metamaterial Panel Change Sound</h1>
            <p class="mv-hero-copy">
                Choose a sound, tune the panel, and hear how bare and metamaterial A4 panels change it.
                All results are simulations.
            </p>
            <p class="mv-hero-note">Headphones recommended. About one minute.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def preferred_resonance_frequency(cached_frequencies: list[float]) -> float:
    if not cached_frequencies:
        return DEFAULT_RESONANCE_HZ
    return min(cached_frequencies, key=lambda value: abs(float(value) - DEFAULT_RESONANCE_HZ))


def build_audio_source_signature(
    source_type: str,
    duration_seconds: float,
    uploaded_file,
) -> tuple[object, ...]:
    if uploaded_file is None:
        return source_type, round(float(duration_seconds), 3)
    return (
        source_type,
        getattr(uploaded_file, "name", None),
        getattr(uploaded_file, "size", None),
        round(float(duration_seconds), 3),
    )


def render_demo_controls(
    cache_points: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    recorded_audio = discover_recorded_audio_files()
    audio_options = [*recorded_audio, UPLOAD_AUDIO_LABEL]
    with st.container(border=True, key="listening_stage"):
        st.markdown(
            """
            <h2 class="mv-stage-title">Create a sound comparison</h2>
            <p class="mv-stage-copy">
                Choose a recording, then set the panel's target frequency.
            </p>
            """,
            unsafe_allow_html=True,
        )
        source_col, target_col = st.columns(2, gap="large")
        with source_col:
            source_type = st.selectbox(
                "Choose a sound",
                options=audio_options,
                index=0,
                key="audio_source",
                help="Included recordings are ready to compare. Choose the upload option only if you want to use your own WAV file.",
            )
            if source_type != UPLOAD_AUDIO_LABEL:
                st.caption("A curated 8-second excerpt will be used.")

        with target_col:
            cached_frequencies = cached_resonance_frequencies(cache_points)
            default_resonance = preferred_resonance_frequency(cached_frequencies)
            if cached_frequencies:
                resonance_hz = st.select_slider(
                    "Target frequency",
                    options=cached_frequencies,
                    value=default_resonance,
                    format_func=lambda value: f"{float(value):.0f} Hz",
                    key="resonance_hz",
                    help="The resonators are tuned to reduce sound energy around this frequency.",
                )
            else:
                resonance_hz = st.slider(
                    "Target frequency",
                    100.0,
                    2000.0,
                    float(default_resonance),
                    10.0,
                    key="resonance_hz",
                    help="The resonators are tuned to reduce sound energy around this frequency.",
                )
            st.caption(f"The panel targets sound energy around {float(resonance_hz):.0f} Hz.")

        uploaded_file = None
        duration_seconds = 8.0
        if source_type == UPLOAD_AUDIO_LABEL:
            uploaded_file = st.file_uploader(
                "Upload a WAV recording",
                type=["wav"],
                key="audio_upload",
                help="Maximum size: 20 MB.",
            )
            duration_seconds = st.slider(
                "Excerpt duration",
                5.0,
                10.0,
                8.0,
                0.5,
                key="upload_duration",
                format="%.1f seconds",
            )
            st.caption(
                "Your WAV is processed in memory for this browser session and is not intentionally saved. "
                "Do not upload confidential recordings."
            )

        run_comparison = st.button(
            "Create listening comparison",
            type="primary",
            width="stretch",
            key="run_comparison",
            disabled=source_type == UPLOAD_AUDIO_LABEL and uploaded_file is None,
        )

    values = build_demo_values(float(resonance_hz))
    request = {
        "source_type": source_type,
        "source_path": recorded_audio.get(source_type),
        "uploaded_file": uploaded_file,
        "duration_seconds": float(duration_seconds),
        "run_comparison": bool(run_comparison),
        "source_signature": build_audio_source_signature(
            source_type,
            duration_seconds,
            uploaded_file,
        ),
    }
    return values, request


def render_auralization(
    infinite,
    finite_results: dict[str, object],
    bare_results: dict[str, object],
    filter_signature: tuple,
    audio_request: dict[str, object],
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    fem_lrm_curves = fem_lrm_curves or {}
    result_signature = (filter_signature, audio_request["source_signature"])
    stored_result = st.session_state.get("window_auralization")
    if stored_result is not None and stored_result.get("result_signature") != result_signature:
        st.session_state.pop("window_auralization", None)

    if bool(audio_request["run_comparison"]):
        st.session_state.pop("window_auralization", None)
        try:
            audio_signal, sample_rate = load_requested_audio(audio_request)
            with st.spinner("Creating the listening comparison…"):
                infinite_audio, bare_audio, finite_audio, fem_audio = auralize_window_filters(
                    audio_signal,
                    sample_rate,
                    infinite,
                    finite_results,
                    bare_results,
                    fem_lrm_curves,
                )

            primary_signals: dict[str, np.ndarray] = {
                ORIGINAL_AUDIO_LABEL: np.asarray(
                    infinite_audio.input_signal,
                    dtype=np.float64,
                ),
            }
            for paper_name, result in bare_audio.items():
                primary_signals[build_bare_audio_output_label(paper_name)] = np.asarray(
                    result.output_signal,
                    dtype=np.float64,
                )
            for paper_name, result in finite_audio.items():
                primary_signals[build_audio_output_label(paper_name)] = np.asarray(
                    result.output_signal,
                    dtype=np.float64,
                )
            for paper_name, result in fem_audio.items():
                primary_signals[fem_curve_display_label(paper_name)] = np.asarray(
                    result.output_signal,
                    dtype=np.float64,
                )
            infinite_signal = np.asarray(infinite_audio.output_signal, dtype=np.float64)
            all_signals = [*primary_signals.values(), infinite_signal]
            if any(signal.size < 2 or not np.all(np.isfinite(signal)) for signal in all_signals):
                raise ValueError("The processed audio contains no usable samples.")

            level_deltas = build_level_deltas(primary_signals)
            psychoacoustic_metrics: dict[str, TrackPsychoacoustics] = {}
            psychoacoustic_error: str | None = None
            try:
                psychoacoustic_metrics = compute_psychoacoustic_comparison(
                    primary_signals,
                    int(infinite_audio.sample_rate),
                )
            except (NativePsychoacousticError, OSError, ValueError, TypeError):
                psychoacoustic_error = (
                    "Sharpness and tonality could not be calculated for this comparison."
                )
            st.session_state["window_auralization"] = {
                "primary_signals": primary_signals,
                "infinite_signal": infinite_signal,
                "level_deltas": level_deltas,
                "psychoacoustic_metrics": psychoacoustic_metrics,
                "psychoacoustic_error": psychoacoustic_error,
                "sample_rate": int(infinite_audio.sample_rate),
                "debug": infinite_audio.sample_rate_debug,
                "result_signature": result_signature,
            }
        except (EOFError, OSError, ValueError, TypeError):
            st.error(
                "We could not create this comparison. If you uploaded a file, try a standard "
                "PCM or floating-point WAV and make sure it contains audible samples."
            )
        except Exception:
            st.error("The listening comparison is temporarily unavailable. Please try again.")

    result_state = st.session_state.get("window_auralization")
    if result_state is None:
        st.caption("Your listening comparison will appear here after processing.")
        return

    signals = result_state["primary_signals"]
    level_deltas = result_state["level_deltas"]
    summary = build_listening_summary(level_deltas)
    st.markdown(
        f"""
        <div class="mv-result-lede" role="status">
            <strong>What changed in this simulation</strong>
            <p>{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample_rate = int(result_state["sample_rate"])
    descriptions = {
        ORIGINAL_AUDIO_LABEL: "The unprocessed recording used as the listening reference.",
        BARE_PANEL_AUDIO_LABEL: "The panel itself without LRM.",
        ANALYTICAL_PANEL_AUDIO_LABEL: (
            "An analytical prediction for the A4 panel with LRM."
        ),
        FEM_PANEL_AUDIO_LABEL: (
            "A finite-element prediction for the same metamaterial panel and tuning."
        ),
    }
    render_audio_comparison(
        signals,
        sample_rate,
        level_deltas_db=level_deltas,
        descriptions=descriptions,
        key="metavision_primary_audio_comparison",
    )
    st.caption(
        "The player keeps the same time position when you switch versions. Level differences are "
        "relative computer predictions, not calibrated sound-pressure measurements."
    )

    with st.expander("Additional model detail", expanded=False):
        st.write(
            "The original technical view compares the source, ideal infinite-panel model, "
            "finite A4 analytical model, and FEM model. Each card combines a waveform, a "
            "log-frequency spectrogram, and its own audio player."
        )
        show_diagnostics = st.toggle(
            "Show four-track waveform and spectrogram comparison",
            value=False,
            key="show_audio_diagnostics",
        )
        if show_diagnostics:
            diagnostic_signals = normalize_audio_group(
                build_diagnostic_signals(
                    signals,
                    result_state["infinite_signal"],
                )
            )
            diagnostic_deltas = build_level_deltas(diagnostic_signals)
            diagnostic_colors = {
                ORIGINAL_AUDIO_LABEL: "#475467",
                INFINITE_PANEL_AUDIO_LABEL: COLOR_INFINITE,
                ANALYTICAL_PANEL_AUDIO_LABEL: COLOR_ANALYTICAL,
                FEM_PANEL_AUDIO_LABEL: COLOR_FEM,
            }
            diagnostic_items = list(diagnostic_signals.items())
            for row_start in range(0, len(diagnostic_items), 2):
                columns = st.columns(2, gap="large")
                for column_index, (label, signal) in enumerate(
                    diagnostic_items[row_start : row_start + 2]
                ):
                    index = row_start + column_index
                    with columns[column_index]:
                        with st.container(border=True):
                            delta = diagnostic_deltas[label]
                            level_description = (
                                "Reference level"
                                if label == ORIGINAL_AUDIO_LABEL
                                else describe_level_delta(delta)
                            )
                            st.markdown(
                                f'<div class="audio-card-title">{label}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="audio-card-caption">{level_description}</div>',
                                unsafe_allow_html=True,
                            )
                            st.plotly_chart(
                                build_audio_card_figure(
                                    signal,
                                    sample_rate,
                                    diagnostic_colors[label],
                                ),
                                width="stretch",
                                config={"displayModeBar": False},
                                key=f"technical_audio_plot_{index}",
                            )
                            st.audio(signal, sample_rate=sample_rate, format="audio/wav")
            debug = result_state["debug"]
            st.caption(
                f"Processed at {debug.processing_rate} Hz "
                f"({debug.selection_mode}; input {debug.input_rate} Hz; "
                f"response to {debug.max_frequency:.1f} Hz)."
            )


def load_requested_audio(audio_request: dict[str, object]) -> tuple[np.ndarray, int]:
    source_type = str(audio_request["source_type"])
    duration_seconds = float(audio_request["duration_seconds"])
    if source_type == UPLOAD_AUDIO_LABEL:
        uploaded_file = audio_request.get("uploaded_file")
        if uploaded_file is None:
            raise ValueError("No WAV file was uploaded.")
        audio_signal, sample_rate = decode_uploaded_wav(uploaded_file)
    else:
        source_path = audio_request.get("source_path")
        if source_path is None:
            raise ValueError("The selected example recording is unavailable.")
        audio_signal, sample_rate = load_wav_file(Path(source_path))

    if sample_rate <= 0:
        raise ValueError("The WAV sample rate must be positive.")
    audio_signal = trim_audio(audio_signal, sample_rate, duration_seconds)
    if audio_signal.size < 2:
        raise ValueError("The WAV file is empty.")
    if not np.all(np.isfinite(audio_signal)):
        raise ValueError("The WAV file contains invalid sample values.")
    if float(np.max(np.abs(audio_signal))) < 1e-8:
        raise ValueError("The WAV file is too quiet to compare.")
    return audio_signal, int(sample_rate)


def build_level_deltas(signals: dict[str, np.ndarray]) -> dict[str, float]:
    if ORIGINAL_AUDIO_LABEL not in signals:
        raise ValueError("An Original reference track is required.")
    original_level = audio_level_db(signals[ORIGINAL_AUDIO_LABEL])
    return {
        label: float(audio_level_db(signal) - original_level)
        for label, signal in signals.items()
    }


def describe_level_delta(delta_db: float) -> str:
    delta_db = float(delta_db)
    if abs(delta_db) < 0.1:
        return "about the same overall level as the original"
    direction = "quieter" if delta_db < 0.0 else "louder"
    return f"{abs(delta_db):.1f} dB {direction} overall than the original"


def build_listening_summary(level_deltas: dict[str, float]) -> str:
    statements: list[str] = []
    if BARE_PANEL_AUDIO_LABEL in level_deltas:
        statements.append(
            "The bare panel is "
            f"{describe_level_delta(level_deltas[BARE_PANEL_AUDIO_LABEL])}."
        )
    if ANALYTICAL_PANEL_AUDIO_LABEL in level_deltas:
        statements.append(
            "With local resonators, the analytical version is "
            f"{describe_level_delta(level_deltas[ANALYTICAL_PANEL_AUDIO_LABEL])}."
        )
    if FEM_PANEL_AUDIO_LABEL in level_deltas:
        statements.append(
            "The FEM version is "
            f"{describe_level_delta(level_deltas[FEM_PANEL_AUDIO_LABEL])}."
        )
    else:
        statements.append("A matching FEM result is not available for this target frequency.")
    statements.append("Switch versions while playback continues to compare the same moment.")
    return " ".join(statements)


def build_diagnostic_signals(
    primary_signals: dict[str, np.ndarray],
    infinite_signal: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the four tracks used by the original technical comparison."""

    diagnostics: dict[str, np.ndarray] = {}
    if ORIGINAL_AUDIO_LABEL in primary_signals:
        diagnostics[ORIGINAL_AUDIO_LABEL] = primary_signals[ORIGINAL_AUDIO_LABEL]
    diagnostics[INFINITE_PANEL_AUDIO_LABEL] = np.asarray(
        infinite_signal,
        dtype=np.float64,
    )
    for label in (ANALYTICAL_PANEL_AUDIO_LABEL, FEM_PANEL_AUDIO_LABEL):
        if label in primary_signals:
            diagnostics[label] = primary_signals[label]
    return diagnostics


def interpolate_stl_at_frequency(
    frequencies_hz: np.ndarray,
    stl_db: np.ndarray,
    target_hz: float,
) -> float | None:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    values = np.asarray(stl_db, dtype=np.float64)
    valid = np.isfinite(frequencies) & np.isfinite(values) & (frequencies > 0.0)
    if not np.any(valid):
        return None
    frequencies = frequencies[valid]
    values = values[valid]
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    values = values[order]
    if target_hz < frequencies[0] or target_hz > frequencies[-1]:
        return None
    return float(np.interp(float(target_hz), frequencies, values))


def build_psychoacoustic_table(
    metrics: dict[str, TrackPsychoacoustics],
) -> pd.DataFrame:
    """Build a stable, listening-order table of calculated sound-quality metrics."""

    rows = []
    for label in PRIMARY_AUDIO_LABELS:
        result = metrics.get(label)
        if result is None:
            continue
        rows.append(
            {
                "Listening version": label,
                "Sharpness S₅ (acum)": result.sharpness_s5_acum,
                "Tonality K₅ (t.u.)": result.tonality_k5_tu,
            }
        )
    return pd.DataFrame(rows)


def build_psychoacoustic_figure(
    metrics: dict[str, TrackPsychoacoustics],
) -> go.Figure:
    """Compare sharpness and tonality with separate, readable horizontal scales."""

    ordered = [
        (label, metrics[label])
        for label in reversed(PRIMARY_AUDIO_LABELS)
        if label in metrics
    ]
    labels = [PSYCHOACOUSTIC_DISPLAY_LABELS[label] for label, _ in ordered]
    source_labels = [label for label, _ in ordered]
    colors_by_label = {
        ORIGINAL_AUDIO_LABEL: COLOR_INK,
        BARE_PANEL_AUDIO_LABEL: COLOR_INFINITE,
        ANALYTICAL_PANEL_AUDIO_LABEL: COLOR_ANALYTICAL,
        FEM_PANEL_AUDIO_LABEL: COLOR_FEM,
    }
    colors = [colors_by_label[label] for label, _ in ordered]
    sharpness = [result.sharpness_s5_acum for _, result in ordered]
    tonality = [result.tonality_k5_tu for _, result in ordered]

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.29,
        subplot_titles=(
            "<b>Sharpness S₅</b><br><span style='font-size:11px'>"
            "Higher can sound brighter or hissier (acum)</span>",
            "<b>Tonality K₅</b><br><span style='font-size:11px'>"
            "Higher means a tone stands out more (t.u.)</span>",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=sharpness,
            y=labels,
            orientation="h",
            marker={"color": colors},
            customdata=source_labels,
            text=[f"{value:.3f}" for value in sharpness],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{customdata}<br>Sharpness: %{x:.3f} acum<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=tonality,
            y=labels,
            orientation="h",
            marker={"color": colors},
            customdata=source_labels,
            text=[f"{value:.3f}" for value in tonality],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{customdata}<br>Tonality: %{x:.3f} t.u.<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for row, values in ((1, sharpness), (2, tonality)):
        axis_max = max(values, default=1.0)
        fig.update_xaxes(
            range=[0.0, axis_max * 1.2],
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            fixedrange=True,
            tickfont={"size": 10, "color": COLOR_MUTED},
            row=row,
            col=1,
        )
        fig.update_yaxes(
            fixedrange=True,
            tickfont={"size": 11, "color": COLOR_INK},
            row=row,
            col=1,
        )
    fig.update_annotations(
        x=0.0,
        xanchor="left",
        align="left",
        font={"size": 14, "color": COLOR_INK},
    )
    fig.update_layout(
        height=570,
        margin={"l": 132, "r": 46, "t": 78, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_MUTED, "size": 11},
        showlegend=False,
        hovermode="closest",
        bargap=0.32,
    )
    return fig


def build_model_snapshot_html(
    resonance_hz: float,
    analytical_value: float | None,
    fem_value: float | None,
    infinite_value: float | None,
) -> str:
    """Return a compact evidence strip for the selected target frequency."""

    facts = [("Target frequency", f"{resonance_hz:.0f} Hz")]
    if analytical_value is not None:
        facts.append(("Metamaterial, analytical", f"{analytical_value:.1f} dB"))
    if fem_value is not None:
        facts.append(("Metamaterial, FEM", f"{fem_value:.1f} dB"))
    if infinite_value is not None:
        facts.append(("Infinite panel", f"{infinite_value:.1f} dB"))
    cells = "".join(
        f'<div class="mv-model-fact" role="listitem"><span>{label}</span>'
        f"<strong>{value}</strong></div>"
        for label, value in facts
    )
    return (
        '<div class="mv-model-facts" role="list" '
        f'aria-label="Model values at the selected target frequency">{cells}</div>'
    )


def build_psychoacoustic_takeaway(
    metrics: dict[str, TrackPsychoacoustics],
) -> str:
    """Summarize the strongest sharpness and tonality sensations in plain language."""

    ordered_metrics = [
        (label, metrics[label]) for label in PRIMARY_AUDIO_LABELS if label in metrics
    ]
    if not ordered_metrics:
        return ""
    sharpest_label, _ = max(
        ordered_metrics,
        key=lambda item: item[1].sharpness_s5_acum,
    )
    most_tonal_label, _ = max(
        ordered_metrics,
        key=lambda item: item[1].tonality_k5_tu,
    )
    return (
        f"Highest sharpness: **{sharpest_label}**. "
        f"Highest tonality: **{most_tonal_label}**."
    )


def render_evidence_section(
    infinite,
    finite_results: dict[str, object],
    resonance_hz: float,
    fem_lrm_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    cache_error: str | None = None,
) -> None:
    st.divider()
    st.markdown(
        """
        <div>
            <h2>Why do the simulations sound different?</h2>
            <p class="mv-evidence-copy">
                The chart predicts how much sound each model stops at every frequency. Higher values mean
                less sound passes through. The analytical model is simplified; Finite Element Method (FEM)
                resolves the panel numerically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    finite_result = finite_results.get(DEMO_WINDOW_NAME)
    if finite_result is None and finite_results:
        finite_result = next(iter(finite_results.values()))
    analytical_value = (
        interpolate_stl_at_frequency(
            finite_result.freqs_hz,
            finite_result.stl_db,
            resonance_hz,
        )
        if finite_result is not None
        else None
    )
    infinite_value = interpolate_stl_at_frequency(
        infinite.freqs_hz,
        infinite.stl_db,
        resonance_hz,
    )
    fem_curve = fem_lrm_curves.get(FEM_CACHE_CURVE_KEY)
    fem_value = (
        interpolate_stl_at_frequency(fem_curve[0], fem_curve[1], resonance_hz)
        if fem_curve is not None
        else None
    )

    st.html(
        build_model_snapshot_html(
            resonance_hz,
            analytical_value,
            fem_value,
            infinite_value,
        )
    )

    if fem_curve is None:
        st.warning(
            "A matching FEM result is not available for this frequency, so this view shows the "
            "analytical predictions only."
        )
    st.plotly_chart(
        build_plot(
            infinite,
            finite_results,
            resonance_hz,
            fem_lrm_curves,
            f_min_hz=DEFAULT_F_MIN_HZ,
            f_max_hz=DEFAULT_F_MAX_HZ,
        ),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="model_evidence_plot",
    )
    st.caption(
        "Sound transmission loss is a model output in decibels. It should not be read as the "
        "sound level a listener would measure in a specific room."
    )

    st.subheader("What your ears may notice")
    result_state = st.session_state.get("window_auralization") or {}
    psychoacoustic_metrics = result_state.get("psychoacoustic_metrics") or {}
    psychoacoustic_error = result_state.get("psychoacoustic_error")
    if psychoacoustic_metrics:
        st.markdown(
            f"**At a glance:** {build_psychoacoustic_takeaway(psychoacoustic_metrics)}"
        )
        st.plotly_chart(
            build_psychoacoustic_figure(psychoacoustic_metrics),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="psychoacoustic_comparison_plot",
        )
        st.caption(
            "Each panel has its own scale. Compare bars within sharpness or within tonality, not "
            "across the two metrics. S₅ and K₅ are calculated before playback normalization."
        )
    elif psychoacoustic_error:
        st.info(psychoacoustic_error)
    else:
        st.caption(
            "Create a listening comparison to calculate sharpness and tonality for each version."
        )

    sharpness_col, tonality_col = st.columns(2, gap="large")
    with sharpness_col:
        st.markdown(
            """
            **Sharpness: listen for brightness or hiss.**

            Higher S₅ can mean more brightness or hiss. DIN 45692 calculates it from
            time-varying ISO 532-1 specific loudness.
            """
        )
    with tonality_col:
        st.markdown(
            """
            **Tonality: listen for a hum, whine, or distinct pitch.**

            Higher K₅ means a tone stands out more strongly from the surrounding sound.
            Values use the Aures tonality model.
            """
        )

    with st.expander("Model assumptions and technical details", expanded=False):
        width_m, height_m = PAPER_WINDOW_SIZES_M[DEMO_WINDOW_NAME]
        st.markdown(
            f"""
            - **Panel size:** A4 ({width_m * 1000:.0f} × {height_m * 1000:.0f} mm)
            - **Host panel:** {FIXED_THICKNESS_M * 1000:.1f} mm aluminium plate
            - **Resonator mass ratio:** {FIXED_MASS_RATIO * 100:.0f}%
            - **Resonator loss factor:** {FIXED_RESONATOR_LOSS_FACTOR:.2f}
            - **Incidence model:** diffuse-field average
            - **Bare listening track:** finite A4 host panel with no local resonators
            - **Analytical curves:** infinite-panel reference plus a finite A4 correction
            - **FEM curve:** cached Siemens Simcenter 3D SOL 108 finite-element result for the selected tuning
            - **Psychoacoustic convention:** free-field model, treating 1.0 full scale as 1 Pa (94 dB SPL)

            Every processed listening version applies its modeled transmission response to the same source recording.
            Uploaded recordings are not calibrated measurements. Perception still depends on playback level,
            headphones, and listener.
            """
        )
        if cache_error:
            st.caption("The FEM cache could not be read during this session.")


def render_project_info_section() -> None:
    st.divider()
    st.subheader("About this demonstration")
    st.write(
        "This public listening demo was created within METAVISION, a Horizon Europe Doctoral "
        "Network studying metamaterials for vibration and sound reduction."
    )
    project_col, privacy_col = st.columns(2, gap="large")
    with project_col:
        st.markdown(
            f"""
            **Research and contact**

            {ABOUT_AUTHOR}

            [{ABOUT_CONTACT}](mailto:{ABOUT_CONTACT})
            """
        )
    with privacy_col:
        st.markdown(
            """
            **If you upload audio**

            The file is processed in memory for this browser session and is not intentionally saved by
            the application. Do not upload confidential or personally identifying recordings.
            """
        )

    with st.expander("Funding and acknowledgements", expanded=False):
        st.markdown(
            ABOUT_ACKNOWLEDGEMENTS
        )
    with st.expander("References", expanded=False):
        st.markdown(
            "\n".join(
                f"- [{item['label']}]({item['url']})"
                for item in REFERENCE_ITEMS
            )
        )

    partner_markup: list[str] = []
    for logo in BRAND_LOGOS:
        label = str(logo["label"])
        logo_uri = resource_data_uri(str(logo["relative_path"]))
        image_markup = f'<img src="{logo_uri}" alt="{label} logo">' if logo_uri else ""
        partner_markup.append(
            f'<div class="mv-partner">{image_markup}</div>'
        )
    st.markdown(
        '<div class="mv-partner-strip" aria-label="Project partners">'
        + "".join(partner_markup)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mv-footer-note">Explore more demonstrations at '
        f'<a href="{METAVISION_DEMOS_URL}">METAVISION</a>.</p>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Hear How a metamaterial panel change sound | METAVISION",
        page_icon=PROJECT_ROOT / "archive/metavision.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_demo_styles()
    render_brand_rail()
    render_hero()

    fem_lookup = prepare_fem_cache_lookup()
    cache_points, cache_error = load_fem_cache_points()
    values, audio_request = render_demo_controls(cache_points)
    show_fem_cache = bool(values.pop("show_fem_cache", False))
    values.pop("show_fem_lrm", False)
    try:
        with st.spinner("Updating the panel model…"):
            infinite, finite_results, bare_results = compute_curves(**values)
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

    render_auralization(
        infinite,
        finite_results,
        bare_results,
        build_filter_signature(
            values,
            infinite,
            finite_results,
            bare_results,
            fem_overlay_curves,
        ),
        audio_request,
        fem_overlay_curves,
    )
    render_evidence_section(
        infinite,
        finite_results,
        float(values["resonance_hz"]),
        fem_overlay_curves,
        cache_error,
    )
    render_project_info_section()


if __name__ == "__main__":
    main()
