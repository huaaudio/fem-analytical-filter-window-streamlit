from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from secondment.materials import available_materials, get_material


PAPER_WINDOW_SIZES_M = {
    "A0": (0.841, 1.189),
    "A1": (0.594, 0.841),
    "A2": (0.420, 0.594),
    "A3": (0.297, 0.420),
    "A4": (0.210, 0.297),
}
DEFAULT_PAPER_WINDOW_NAMES = tuple(PAPER_WINDOW_SIZES_M)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    f_min_hz: float = 10.0
    f_max_hz: float = 4000.0
    df_hz: float = 5.0
    plate_type: str = "meta"
    material: str | None = None
    density: float = 2700.0
    thickness_m: float = 0.002
    young_modulus: float = 70e9
    poisson_ratio: float = 0.3
    host_loss_factor: float = 0.02
    resonance_hz: float = 950.0
    mass_ratio: float = 0.20
    resonator_loss_factor: float = 0.0
    unit_cell_a_m: float = 0.04
    unit_cell_b_m: float = 0.038
    partition_type: str = "single"
    cavity_thickness_m: float = 0.05
    second_material: str | None = None
    second_thickness_m: float | None = None
    second_plate_type: str | None = None
    paper_windows: tuple[str, ...] = DEFAULT_PAPER_WINDOW_NAMES
    incidence: str = "diffuse"
    theta_oblique_deg: float = 0.0
    theta_limit_deg: float = 90.0
    theta_samples: int = 181
    radial_samples: int = 600
    output_dir: Path = Path("results/analysis")
    output_stem: str = "analytical_filter_window_comparison"


def _frequency_axis(config: ComparisonConfig) -> np.ndarray:
    if config.f_min_hz <= 0.0:
        raise ValueError("f_min_hz must be positive.")
    if config.f_max_hz <= config.f_min_hz:
        raise ValueError("f_max_hz must be greater than f_min_hz.")
    if config.df_hz <= 0.0:
        raise ValueError("df_hz must be positive.")
    return np.arange(config.f_min_hz, config.f_max_hz + 0.5 * config.df_hz, config.df_hz)


def _build_host(config: ComparisonConfig) -> HostPanel:
    if config.material is not None:
        return get_material(config.material).host_panel(
            config.thickness_m, loss_factor=config.host_loss_factor
        )
    return HostPanel(
        density=config.density,
        thickness=config.thickness_m,
        young_modulus=config.young_modulus,
        poisson_ratio=config.poisson_ratio,
        loss_factor=config.host_loss_factor,
    )


def _build_resonator(config: ComparisonConfig) -> LocalResonator | None:
    if config.plate_type == "bare":
        return None
    if config.plate_type != "meta":
        raise ValueError("plate_type must be 'bare' or 'meta'.")
    return LocalResonator(
        resonance_frequency_hz=config.resonance_hz,
        mass_ratio=config.mass_ratio,
        unit_cell_area=config.unit_cell_a_m * config.unit_cell_b_m,
        stiffness_loss_factor=config.resonator_loss_factor,
    )


def _build_second_leaf(config: ComparisonConfig) -> PartitionLeaf:
    material = config.second_material if config.second_material is not None else config.material
    thickness = (
        config.second_thickness_m if config.second_thickness_m is not None else config.thickness_m
    )
    if material is not None:
        host = get_material(material).host_panel(thickness, loss_factor=config.host_loss_factor)
    else:
        host = HostPanel(
            density=config.density,
            thickness=thickness,
            young_modulus=config.young_modulus,
            poisson_ratio=config.poisson_ratio,
            loss_factor=config.host_loss_factor,
        )
    plate_type = config.second_plate_type if config.second_plate_type is not None else config.plate_type
    resonator = None
    if plate_type == "meta":
        resonator = LocalResonator(
            resonance_frequency_hz=config.resonance_hz,
            mass_ratio=config.mass_ratio,
            unit_cell_area=config.unit_cell_a_m * config.unit_cell_b_m,
            stiffness_loss_factor=config.resonator_loss_factor,
        )
    elif plate_type != "bare":
        raise ValueError("second_plate_type must be 'bare' or 'meta'.")
    return PartitionLeaf(host=host, resonator=resonator)



def paper_window_area_m2(paper_name: str) -> float:
    try:
        side_a, side_b = PAPER_WINDOW_SIZES_M[paper_name]
    except KeyError as exc:
        supported = ", ".join(PAPER_WINDOW_SIZES_M)
        raise ValueError(f"Unsupported paper window '{paper_name}'. Choose one of: {supported}.") from exc
    return side_a * side_b


def paper_window_label(paper_name: str) -> str:
    side_a, side_b = PAPER_WINDOW_SIZES_M[paper_name]
    return f"{paper_name} ({side_a:g} m x {side_b:g} m)"


def normalized_paper_windows(config: ComparisonConfig) -> tuple[str, ...]:
    names = tuple(config.paper_windows)
    if not names:
        raise ValueError("At least one paper window size is required.")
    for name in names:
        paper_window_area_m2(name)
    return names


def _write_csv(path: Path, infinite_result, finite_results) -> None:
    header = [
        "freq_hz",
        "infinite_pressure_frf",
        "infinite_tau_power",
        "infinite_stl_db",
    ]
    for paper_name in finite_results:
        header.extend(
            [
                f"{paper_name}_pressure_frf",
                f"{paper_name}_tau_power",
                f"{paper_name}_stl_db",
                f"{paper_name}_minus_infinite_stl_db",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(infinite_result.freqs_hz.size):
            row = [
                infinite_result.freqs_hz[index],
                infinite_result.pressure_frf[index],
                infinite_result.power_tau[index],
                infinite_result.stl_db[index],
            ]
            for finite_result in finite_results.values():
                row.extend(
                    [
                        finite_result.pressure_frf[index],
                        finite_result.power_tau[index],
                        finite_result.stl_db[index],
                        finite_result.stl_db[index] - infinite_result.stl_db[index],
                    ]
                )
            writer.writerow([float(np.real(value)) for value in row])


def _write_plot(path: Path, infinite_result, finite_results, config: ComparisonConfig) -> None:
    plt.figure(figsize=(9.0, 5.2))
    plt.semilogx(
        infinite_result.freqs_hz,
        infinite_result.stl_db,
        label="Infinite panel",
        linewidth=2.0,
    )
    for paper_name, finite_result in finite_results.items():
        plt.semilogx(
            finite_result.freqs_hz,
            finite_result.stl_db,
            label=paper_window_label(paper_name),
            linewidth=1.8,
        )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("STL [dB]")
    partition_label = "double-leaf" if config.partition_type == "double" else "single-leaf"
    material_label = f", {config.material}" if config.material else ""
    plt.title(
        f"{config.plate_type.capitalize()} {partition_label}{material_label} "
        "analytical filter: infinite vs A0-A4 windows"
    )
    plt.grid(True, which="both", alpha=0.28)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_comparison(config: ComparisonConfig) -> dict[str, Path]:
    freqs = _frequency_axis(config)
    host = _build_host(config)
    resonator = _build_resonator(config)
    paper_windows = normalized_paper_windows(config)

    if config.partition_type not in {"single", "double"}:
        raise ValueError("partition_type must be 'single' or 'double'.")

    common_kwargs = {
        "incidence": config.incidence,
        "theta_oblique_rad": np.deg2rad(config.theta_oblique_deg),
        "theta_limit_rad": np.deg2rad(config.theta_limit_deg),
        "theta_samples": config.theta_samples,
        "radial_samples": config.radial_samples,
    }

    if config.partition_type == "single":
        def filter_call(size_type: str, **window_kwargs):
            return bending_panel_pressure_filter(
                freqs,
                host=host,
                resonator=resonator,
                size_type=size_type,
                **window_kwargs,
                **common_kwargs,
            )
    else:
        if config.cavity_thickness_m <= 0.0:
            raise ValueError("cavity_thickness_m must be positive for a double partition.")
        leaves = (PartitionLeaf(host=host, resonator=resonator), _build_second_leaf(config))
        cavities = (AirCavity(thickness_m=config.cavity_thickness_m),)

        def filter_call(size_type: str, **window_kwargs):
            return partition_pressure_filter(
                freqs,
                leaves,
                cavities,
                size_type=size_type,
                **window_kwargs,
                **common_kwargs,
            )

    infinite_result = filter_call("infinite")
    finite_results = {
        paper_name: filter_call(
            "finite",
            window_area_m2=paper_window_area_m2(paper_name),
        )
        for paper_name in paper_windows
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{config.output_stem}.csv"
    png_path = output_dir / f"{config.output_stem}.png"

    _write_csv(csv_path, infinite_result, finite_results)
    _write_plot(png_path, infinite_result, finite_results, config)
    return {"csv": csv_path, "png": png_path}


def parse_args(argv: list[str] | None = None) -> ComparisonConfig:
    defaults = ComparisonConfig()
    parser = argparse.ArgumentParser(
        description="Compare analytical infinite-panel and finite-window transmission filters.",
    )
    parser.add_argument("--f-min", type=float, default=defaults.f_min_hz)
    parser.add_argument("--f-max", type=float, default=defaults.f_max_hz)
    parser.add_argument("--df", type=float, default=defaults.df_hz)
    parser.add_argument("--plate-type", choices=["bare", "meta"], default=defaults.plate_type)
    parser.add_argument(
        "--material",
        choices=list(available_materials()),
        default=defaults.material,
        help="Named material preset; overrides --rho/--young/--nu for the (first) leaf.",
    )
    parser.add_argument("--rho", type=float, default=defaults.density)
    parser.add_argument("--thickness-mm", type=float, default=defaults.thickness_m * 1000.0)
    parser.add_argument("--young", type=float, default=defaults.young_modulus)
    parser.add_argument("--nu", type=float, default=defaults.poisson_ratio)
    parser.add_argument("--host-loss", type=float, default=defaults.host_loss_factor)
    parser.add_argument(
        "--partition-type",
        choices=["single", "double"],
        default=defaults.partition_type,
        help="single leaf, or mass-air-mass double leaf.",
    )
    parser.add_argument(
        "--cavity-thickness-mm",
        type=float,
        default=defaults.cavity_thickness_m * 1000.0,
        help="Air gap thickness for a double partition.",
    )
    parser.add_argument(
        "--second-material",
        choices=list(available_materials()),
        default=defaults.second_material,
        help="Material for the second leaf (defaults to the first leaf's material).",
    )
    parser.add_argument(
        "--second-thickness-mm",
        type=float,
        default=None,
        help="Thickness of the second leaf in mm (defaults to the first leaf's thickness).",
    )
    parser.add_argument(
        "--second-plate-type",
        choices=["bare", "meta"],
        default=defaults.second_plate_type,
        help="bare or meta for the second leaf (defaults to --plate-type).",
    )
    parser.add_argument("--resonance-hz", type=float, default=defaults.resonance_hz)
    parser.add_argument("--mass-ratio", type=float, default=defaults.mass_ratio)
    parser.add_argument("--resonator-loss", type=float, default=defaults.resonator_loss_factor)
    parser.add_argument("--unit-cell-a", type=float, default=defaults.unit_cell_a_m)
    parser.add_argument("--unit-cell-b", type=float, default=defaults.unit_cell_b_m)
    parser.add_argument(
        "--paper-window",
        choices=["all", *PAPER_WINDOW_SIZES_M],
        default="all",
        help="Finite window paper preset. Use 'all' to compare A0 through A4.",
    )
    parser.add_argument("--incidence", choices=["oblique", "diffuse"], default=defaults.incidence)
    parser.add_argument("--theta-oblique-deg", type=float, default=defaults.theta_oblique_deg)
    parser.add_argument("--theta-limit-deg", type=float, default=defaults.theta_limit_deg)
    parser.add_argument("--theta-samples", type=int, default=defaults.theta_samples)
    parser.add_argument("--radial-samples", type=int, default=defaults.radial_samples)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--output-stem", default=defaults.output_stem)

    args = parser.parse_args(argv)
    return ComparisonConfig(
        f_min_hz=args.f_min,
        f_max_hz=args.f_max,
        df_hz=args.df,
        plate_type=args.plate_type,
        material=args.material,
        density=args.rho,
        thickness_m=args.thickness_mm / 1000.0,
        young_modulus=args.young,
        poisson_ratio=args.nu,
        host_loss_factor=args.host_loss,
        resonance_hz=args.resonance_hz,
        mass_ratio=args.mass_ratio,
        resonator_loss_factor=args.resonator_loss,
        unit_cell_a_m=args.unit_cell_a,
        unit_cell_b_m=args.unit_cell_b,
        partition_type=args.partition_type,
        cavity_thickness_m=args.cavity_thickness_mm / 1000.0,
        second_material=args.second_material,
        second_thickness_m=(
            None if args.second_thickness_mm is None else args.second_thickness_mm / 1000.0
        ),
        second_plate_type=args.second_plate_type,
        paper_windows=DEFAULT_PAPER_WINDOW_NAMES if args.paper_window == "all" else (args.paper_window,),
        incidence=args.incidence,
        theta_oblique_deg=args.theta_oblique_deg,
        theta_limit_deg=args.theta_limit_deg,
        theta_samples=args.theta_samples,
        radial_samples=args.radial_samples,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    paths = run_comparison(config)
    print(f"Wrote comparison CSV: {paths['csv']}")
    print(f"Wrote comparison plot: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
