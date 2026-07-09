from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


RHO_0 = 1.225
C_0 = 343.0
Z_0 = RHO_0 * C_0


IncidenceType = Literal["oblique", "diffuse"]
SizeType = Literal["infinite", "finite"]


@dataclass(frozen=True, slots=True)
class AmbientMedium:
    density: float = RHO_0
    sound_speed: float = C_0

    @property
    def impedance(self) -> float:
        return float(self.density * self.sound_speed)


@dataclass(frozen=True, slots=True)
class HostPanel:
    density: float = 2700.0
    thickness: float = 0.002
    young_modulus: float = 70e9
    poisson_ratio: float = 0.3
    loss_factor: float = 0.02

    @property
    def surface_density(self) -> float:
        return float(self.density * self.thickness)

    @property
    def bending_stiffness(self) -> complex:
        numerator = self.young_modulus * (1.0 + 1j * self.loss_factor) * self.thickness**3
        denominator = 12.0 * (1.0 - self.poisson_ratio**2)
        return complex(numerator / denominator)


@dataclass(frozen=True, slots=True)
class LocalResonator:
    resonance_frequency_hz: float = 950.0
    mass_ratio: float = 0.20
    unit_cell_area: float = 0.04 * 0.038
    count_per_cell: int = 1
    participating_mass_fraction: float = 1.0
    stiffness_loss_factor: float = 0.0


@dataclass(frozen=True, slots=True)
class AnalyticalFilterResult:
    freqs_hz: np.ndarray
    pressure_frf: np.ndarray
    power_tau: np.ndarray
    stl_db: np.ndarray
    label: str


def _as_frequency_array(freqs_hz: np.ndarray | list[float]) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError("Frequency vector must not be empty.")
    if np.any(freqs <= 0.0):
        raise ValueError("Frequencies must be positive for analytical transmission filters.")
    return freqs


def _validate_host(host: HostPanel) -> None:
    if host.density <= 0.0:
        raise ValueError("Host panel density must be positive.")
    if host.thickness <= 0.0:
        raise ValueError("Host panel thickness must be positive.")
    if host.young_modulus <= 0.0:
        raise ValueError("Host panel Young's modulus must be positive.")
    if not -1.0 < host.poisson_ratio < 0.5:
        raise ValueError("Host panel Poisson ratio must lie between -1 and 0.5.")


def resonator_mass(host: HostPanel, resonator: LocalResonator) -> float:
    _validate_host(host)
    if resonator.mass_ratio < 0.0:
        raise ValueError("Resonator mass ratio must be non-negative.")
    if resonator.unit_cell_area <= 0.0:
        raise ValueError("Resonator unit-cell area must be positive.")
    if resonator.participating_mass_fraction < 0.0:
        raise ValueError("Resonator participating mass fraction must be non-negative.")
    return float(
        host.surface_density
        * resonator.unit_cell_area
        * resonator.mass_ratio
        * resonator.participating_mass_fraction
    )


def equivalent_density_with_resonator(
    freqs_hz: np.ndarray | list[float],
    host: HostPanel,
    resonator: LocalResonator,
) -> np.ndarray:
    """Equivalent density used by the analytical resonator filter in app.py."""
    freqs = _as_frequency_array(freqs_hz)
    _validate_host(host)
    if resonator.resonance_frequency_hz <= 0.0:
        raise ValueError("Resonator frequency must be positive.")
    if resonator.count_per_cell <= 0:
        raise ValueError("Resonator count per cell must be positive.")

    omega = 2.0 * np.pi * freqs
    omega_res = 2.0 * np.pi * resonator.resonance_frequency_hz
    m_res = resonator_mass(host, resonator)
    k_res = m_res * omega_res**2 * (1.0 + 1j * resonator.stiffness_loss_factor)

    denominator = k_res - omega**2 * m_res
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    numerator = m_res * k_res

    base_density = host.density * (
        1.0 + resonator.mass_ratio * (1.0 - resonator.participating_mass_fraction)
    )
    resonator_term = (
        resonator.count_per_cell
        * numerator
        / denominator
        / (resonator.unit_cell_area * host.thickness)
    )
    return base_density + resonator_term


def normal_incidence_single_wall_pressure_transmission(
    freqs_hz: np.ndarray | list[float],
    density: float | np.ndarray,
    thickness: float,
    ambient: AmbientMedium = AmbientMedium(),
) -> np.ndarray:
    """Pressure transmission coefficient for the app.py single-wall analytical model."""
    freqs = _as_frequency_array(freqs_hz)
    if thickness <= 0.0:
        raise ValueError("Panel thickness must be positive.")

    omega = 2.0 * np.pi * freqs
    density_array = np.asarray(density, dtype=np.complex128)
    numerator = 2.0 * ambient.impedance * omega
    denominator = 1j * 2.0 * ambient.impedance * omega - density_array * thickness * omega**2
    return numerator / denominator


def transmission_loss_from_pressure(pressure_frf: np.ndarray) -> np.ndarray:
    magnitude = np.maximum(np.abs(np.asarray(pressure_frf)), 1e-12)
    return -20.0 * np.log10(magnitude)


def transmission_loss_from_power(power_tau: np.ndarray) -> np.ndarray:
    tau = np.maximum(np.real(np.asarray(power_tau, dtype=np.complex128)), 1e-24)
    return -10.0 * np.log10(tau)


def app_style_single_partition_filter(
    freqs_hz: np.ndarray | list[float],
    host: HostPanel,
    resonator: LocalResonator | None = None,
    ambient: AmbientMedium = AmbientMedium(),
) -> AnalyticalFilterResult:
    """Extracted app.py analytical single-partition filter.

    When a resonator is supplied, the host density is replaced by the equivalent
    density before applying the app's single-wall pressure transmission formula.
    """
    freqs = _as_frequency_array(freqs_hz)
    density = (
        host.density
        if resonator is None
        else equivalent_density_with_resonator(freqs, host, resonator)
    )
    pressure_frf = normal_incidence_single_wall_pressure_transmission(
        freqs,
        density,
        host.thickness,
        ambient=ambient,
    )
    power_tau = np.square(np.abs(pressure_frf))
    label = "App-style analytical filter"
    if resonator is not None:
        label += " with resonator"
    return AnalyticalFilterResult(
        freqs_hz=freqs,
        pressure_frf=pressure_frf,
        power_tau=power_tau,
        stl_db=transmission_loss_from_pressure(pressure_frf),
        label=label,
    )


def bending_wave_number(
    freqs_hz: np.ndarray | list[float],
    host: HostPanel,
    resonator: LocalResonator | None = None,
) -> np.ndarray:
    """Bending wave number for the analytical plate/window filter."""
    freqs = _as_frequency_array(freqs_hz)
    _validate_host(host)

    density = (
        host.density
        if resonator is None
        else equivalent_density_with_resonator(freqs, host, resonator)
    )
    omega = 2.0 * np.pi * freqs
    return np.power(density * host.thickness * omega**2 / host.bending_stiffness, 0.25)


def _infinite_oblique_power_tau(
    omega: np.ndarray,
    acoustic_wavenumber: np.ndarray,
    bending_wavenumber: np.ndarray,
    bending_stiffness: complex,
    theta_rad: float,
    ambient: AmbientMedium,
) -> np.ndarray:
    cos_theta = np.cos(theta_rad)
    forcing_wavenumber = acoustic_wavenumber * np.sin(theta_rad)
    plate_impedance = bending_stiffness / (1j * omega) * (
        forcing_wavenumber**4 - bending_wavenumber**4
    )
    denominator = 1.0 + plate_impedance * cos_theta / (2.0 * ambient.impedance)
    return 1.0 / np.square(np.abs(denominator))


def _finite_oblique_sigma(
    acoustic_wavenumber: float,
    theta_rad: float,
    window_length: float,
    radial_samples: int,
) -> float:
    if acoustic_wavenumber <= 0.0:
        return 0.0
    kr = np.linspace(0.0, acoustic_wavenumber, radial_samples, endpoint=False)
    forcing_wavenumber = acoustic_wavenumber * np.sin(theta_rad)
    x = (kr - forcing_wavenumber) * window_length / 2.0
    sinc_squared = np.square(np.sinc(x / np.pi))
    root = np.sqrt(np.maximum(acoustic_wavenumber**2 - kr**2, 1e-30))
    integral = np.trapezoid(sinc_squared / root, kr)
    return float(window_length * acoustic_wavenumber * integral / (2.0 * np.pi))


def _finite_diffuse_power_tau(
    omega: np.ndarray,
    acoustic_wavenumber: np.ndarray,
    bending_wavenumber: np.ndarray,
    bending_stiffness: complex,
    theta_grid: np.ndarray,
    window_length: float,
    radial_samples: int,
    ambient: AmbientMedium,
) -> np.ndarray:
    tau = np.empty_like(acoustic_wavenumber, dtype=np.float64)
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)

    for index, (omega_i, ka_i, kb_i) in enumerate(
        zip(omega, acoustic_wavenumber, bending_wavenumber, strict=True)
    ):
        kr = np.linspace(0.0, ka_i, radial_samples, endpoint=False)
        forcing_wavenumber = ka_i * sin_theta
        x = (kr[None, :] - forcing_wavenumber[:, None]) * window_length / 2.0
        sinc_squared = np.square(np.sinc(x / np.pi))
        root = np.sqrt(np.maximum(ka_i**2 - kr**2, 1e-30))
        sigma_theta = (
            window_length
            * ka_i
            * np.trapezoid(sinc_squared / root[None, :], kr, axis=1)
            / (2.0 * np.pi)
        )

        plate_impedance = bending_stiffness / (1j * omega_i) * (
            (ka_i * sin_theta) ** 4 - kb_i**4
        )
        denominator = 1.0 + plate_impedance * cos_theta / (2.0 * ambient.impedance)
        tau_theta = 1.0 / np.square(np.abs(denominator))
        numerator = np.trapezoid(
            tau_theta * sigma_theta * sin_theta * np.square(cos_theta),
            theta_grid,
        )
        tau[index] = float(numerator)

    diffuse_denominator = 0.5 * np.sin(theta_grid[-1]) ** 2
    return tau / diffuse_denominator


def bending_panel_pressure_filter(
    freqs_hz: np.ndarray | list[float],
    host: HostPanel,
    resonator: LocalResonator | None = None,
    size_type: SizeType = "infinite",
    incidence: IncidenceType = "diffuse",
    window_area_m2: float | None = None,
    window_length_m: float | None = None,
    theta_oblique_rad: float = 0.0,
    theta_limit_rad: float = 0.5 * np.pi,
    theta_samples: int = 181,
    radial_samples: int = 600,
    ambient: AmbientMedium = AmbientMedium(),
) -> AnalyticalFilterResult:
    """Analytical pressure filter for an infinite panel or a finite window.

    The finite-window correction follows the archived MATLAB implementation:
    a rectangular window is represented by an equivalent length sqrt(area).
    The returned pressure FRF is sqrt(power transmission coefficient).
    """
    freqs = _as_frequency_array(freqs_hz)
    _validate_host(host)
    if incidence not in {"oblique", "diffuse"}:
        raise ValueError("incidence must be 'oblique' or 'diffuse'.")
    if size_type not in {"infinite", "finite"}:
        raise ValueError("size_type must be 'infinite' or 'finite'.")
    if theta_samples < 2:
        raise ValueError("theta_samples must be at least 2.")
    if radial_samples < 4:
        raise ValueError("radial_samples must be at least 4.")
    if not 0.0 < theta_limit_rad <= 0.5 * np.pi:
        raise ValueError("theta_limit_rad must be between 0 and pi/2.")

    if size_type == "finite":
        if window_length_m is None:
            if window_area_m2 is None:
                raise ValueError("Finite filters need window_area_m2 or window_length_m.")
            if window_area_m2 <= 0.0:
                raise ValueError("window_area_m2 must be positive.")
            window_length_m = float(np.sqrt(window_area_m2))
        if window_length_m <= 0.0:
            raise ValueError("window_length_m must be positive.")

    omega = 2.0 * np.pi * freqs
    acoustic_wavenumber = omega / ambient.sound_speed
    bending_wavenumber = bending_wave_number(freqs, host, resonator=resonator)
    bending_stiffness = host.bending_stiffness

    if incidence == "oblique":
        power_tau = _infinite_oblique_power_tau(
            omega,
            acoustic_wavenumber,
            bending_wavenumber,
            bending_stiffness,
            theta_oblique_rad,
            ambient,
        )
        if size_type == "finite":
            sigma = np.array(
                [
                    _finite_oblique_sigma(
                        float(ka),
                        theta_oblique_rad,
                        float(window_length_m),
                        radial_samples,
                    )
                    for ka in acoustic_wavenumber
                ],
                dtype=np.float64,
            )
            power_tau = power_tau * np.cos(theta_oblique_rad) * sigma
    else:
        theta_grid = np.linspace(0.0, theta_limit_rad, theta_samples)
        sin_theta = np.sin(theta_grid)
        cos_theta = np.cos(theta_grid)
        diffuse_denominator = 0.5 * np.sin(theta_limit_rad) ** 2

        if size_type == "infinite":
            ka = acoustic_wavenumber[:, None]
            kb = bending_wavenumber[:, None]
            omega_grid = omega[:, None]
            plate_impedance = bending_stiffness / (1j * omega_grid) * (
                np.power(ka * sin_theta[None, :], 4) - np.power(kb, 4)
            )
            denominator = 1.0 + plate_impedance * cos_theta[None, :] / (2.0 * ambient.impedance)
            tau_theta = 1.0 / np.square(np.abs(denominator))
            numerator = np.trapezoid(tau_theta * sin_theta[None, :] * cos_theta[None, :], theta_grid, axis=1)
            power_tau = numerator / diffuse_denominator
        else:
            power_tau = _finite_diffuse_power_tau(
                omega,
                acoustic_wavenumber,
                bending_wavenumber,
                bending_stiffness,
                theta_grid,
                float(window_length_m),
                radial_samples,
                ambient,
            )

    power_tau = np.maximum(np.real(power_tau), 1e-24)
    pressure_frf = np.sqrt(power_tau)
    resonator_label = "meta" if resonator is not None else "bare"
    label = f"{resonator_label} {size_type} {incidence} analytical filter"
    return AnalyticalFilterResult(
        freqs_hz=freqs,
        pressure_frf=pressure_frf,
        power_tau=power_tau,
        stl_db=transmission_loss_from_power(power_tau),
        label=label,
    )


# ---------------------------------------------------------------------------
# Multi-leaf partitions (transfer-matrix method)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AirCavity:
    """Air gap separating two partition leaves."""

    thickness_m: float = 0.05


@dataclass(frozen=True, slots=True)
class PartitionLeaf:
    """A single panel leaf, optionally carrying a local resonator."""

    host: HostPanel
    resonator: LocalResonator | None = None


def _leaf_transfer_impedance(
    freqs_hz: np.ndarray,
    omega: np.ndarray,
    acoustic_wavenumber: np.ndarray,
    sin_theta: np.ndarray,
    leaf: PartitionLeaf,
) -> np.ndarray:
    """Per-angle plate transfer impedance Z_p(omega, theta) for one leaf.

    Returns a complex array of shape (n_freq, n_theta) that matches the panel
    impedance used by :func:`bending_panel_pressure_filter`.
    """
    bending_wavenumber = bending_wave_number(freqs_hz, leaf.host, resonator=leaf.resonator)
    forcing_wavenumber = acoustic_wavenumber[:, None] * sin_theta[None, :]
    return leaf.host.bending_stiffness / (1j * omega[:, None]) * (
        np.power(forcing_wavenumber, 4) - np.power(bending_wavenumber[:, None], 4)
    )


def _mat_mul(a11, a12, a21, a22, b11, b12, b21, b22):
    return (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )


def _partition_tau_theta(
    freqs_hz: np.ndarray,
    omega: np.ndarray,
    acoustic_wavenumber: np.ndarray,
    theta_grid: np.ndarray,
    leaves: tuple[PartitionLeaf, ...],
    cavities: tuple[AirCavity, ...],
    ambient: AmbientMedium,
) -> np.ndarray:
    """Per-angle power transmission coefficient for a stacked partition.

    Implements the transfer-matrix method on the state vector ``[p, v_n]``: each
    leaf contributes a panel element ``[[1, Z_p], [0, 1]]`` and each air cavity a
    fluid-layer element. A single leaf reduces exactly to the closed-form filter.
    """
    cos_theta = np.cos(theta_grid)
    cos_theta_safe = np.where(np.abs(cos_theta) < 1e-9, 1e-9, cos_theta)
    sin_theta = np.sin(theta_grid)

    fluid_impedance = ambient.impedance / cos_theta_safe  # Zc(theta), shape (n_theta,)
    fluid_impedance = fluid_impedance[None, :]
    normal_wavenumber = acoustic_wavenumber[:, None] * cos_theta_safe[None, :]  # kz

    n_freq = omega.size
    n_theta = theta_grid.size
    t11 = np.ones((n_freq, n_theta), dtype=np.complex128)
    t12 = np.zeros((n_freq, n_theta), dtype=np.complex128)
    t21 = np.zeros((n_freq, n_theta), dtype=np.complex128)
    t22 = np.ones((n_freq, n_theta), dtype=np.complex128)

    for index, leaf in enumerate(leaves):
        plate_impedance = _leaf_transfer_impedance(
            freqs_hz, omega, acoustic_wavenumber, sin_theta, leaf
        )
        t11, t12, t21, t22 = _mat_mul(
            t11, t12, t21, t22,
            1.0, plate_impedance, 0.0, 1.0,
        )
        if index < len(cavities):
            thickness = float(cavities[index].thickness_m)
            phase = normal_wavenumber * thickness
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            g12 = 1j * fluid_impedance * sin_phase
            g21 = 1j * sin_phase / fluid_impedance
            t11, t12, t21, t22 = _mat_mul(
                t11, t12, t21, t22,
                cos_phase, g12, g21, cos_phase,
            )

    transmission = 2.0 / (
        t11 + t12 / fluid_impedance + fluid_impedance * t21 + t22
    )
    return np.square(np.abs(transmission))


def _finite_sigma_theta(
    acoustic_wavenumber: float,
    sin_theta: np.ndarray,
    window_length: float,
    radial_samples: int,
) -> np.ndarray:
    """Finite-window radiation efficiency sigma(theta) at one frequency."""
    if acoustic_wavenumber <= 0.0:
        return np.zeros_like(sin_theta)
    kr = np.linspace(0.0, acoustic_wavenumber, radial_samples, endpoint=False)
    forcing_wavenumber = acoustic_wavenumber * sin_theta
    x = (kr[None, :] - forcing_wavenumber[:, None]) * window_length / 2.0
    sinc_squared = np.square(np.sinc(x / np.pi))
    root = np.sqrt(np.maximum(acoustic_wavenumber**2 - kr**2, 1e-30))
    integral = np.trapezoid(sinc_squared / root[None, :], kr, axis=1)
    return np.asarray(window_length * acoustic_wavenumber * integral / (2.0 * np.pi))


def partition_pressure_filter(
    freqs_hz: np.ndarray | list[float],
    leaves: PartitionLeaf | tuple[PartitionLeaf, ...] | list[PartitionLeaf],
    cavities: AirCavity | tuple[AirCavity, ...] | list[AirCavity] = (),
    *,
    size_type: SizeType = "infinite",
    incidence: IncidenceType = "diffuse",
    window_area_m2: float | None = None,
    window_length_m: float | None = None,
    theta_oblique_rad: float = 0.0,
    theta_limit_rad: float = 0.5 * np.pi,
    theta_samples: int = 181,
    radial_samples: int = 600,
    ambient: AmbientMedium = AmbientMedium(),
) -> AnalyticalFilterResult:
    """Analytical transmission filter for single- or double-leaf partitions.

    ``leaves`` is one :class:`PartitionLeaf` (single wall) or a tuple of leaves
    separated by ``cavities`` air gaps (e.g. two leaves with one cavity for a
    mass-air-mass double wall). A single leaf reproduces
    :func:`bending_panel_pressure_filter`.
    """
    freqs = _as_frequency_array(freqs_hz)
    if isinstance(leaves, PartitionLeaf):
        leaves = (leaves,)
    else:
        leaves = tuple(leaves)
    if isinstance(cavities, AirCavity):
        cavities = (cavities,)
    else:
        cavities = tuple(cavities)

    if len(leaves) == 0:
        raise ValueError("A partition needs at least one leaf.")
    if len(cavities) != max(len(leaves) - 1, 0):
        raise ValueError("A partition needs exactly one cavity between adjacent leaves.")
    for leaf in leaves:
        _validate_host(leaf.host)
    for cavity in cavities:
        if cavity.thickness_m <= 0.0:
            raise ValueError("Air cavity thickness must be positive.")
    if incidence not in {"oblique", "diffuse"}:
        raise ValueError("incidence must be 'oblique' or 'diffuse'.")
    if size_type not in {"infinite", "finite"}:
        raise ValueError("size_type must be 'infinite' or 'finite'.")
    if theta_samples < 2:
        raise ValueError("theta_samples must be at least 2.")
    if radial_samples < 4:
        raise ValueError("radial_samples must be at least 4.")
    if not 0.0 < theta_limit_rad <= 0.5 * np.pi:
        raise ValueError("theta_limit_rad must be between 0 and pi/2.")

    if size_type == "finite":
        if window_length_m is None:
            if window_area_m2 is None:
                raise ValueError("Finite filters need window_area_m2 or window_length_m.")
            if window_area_m2 <= 0.0:
                raise ValueError("window_area_m2 must be positive.")
            window_length_m = float(np.sqrt(window_area_m2))
        if window_length_m <= 0.0:
            raise ValueError("window_length_m must be positive.")
    window_length = 0.0 if window_length_m is None else float(window_length_m)

    omega = 2.0 * np.pi * freqs
    acoustic_wavenumber = omega / ambient.sound_speed

    if incidence == "oblique":
        theta_grid = np.array([theta_oblique_rad], dtype=np.float64)
        power_tau = _partition_tau_theta(
            freqs, omega, acoustic_wavenumber, theta_grid, leaves, cavities, ambient
        )[:, 0]
        if size_type == "finite":
            sigma = np.array(
                [
                    _finite_sigma_theta(
                        float(ka),
                        np.array([np.sin(theta_oblique_rad)]),
                        window_length,
                        radial_samples,
                    )[0]
                    for ka in acoustic_wavenumber
                ],
                dtype=np.float64,
            )
            power_tau = power_tau * np.cos(theta_oblique_rad) * sigma
    else:
        theta_grid = np.linspace(0.0, theta_limit_rad, theta_samples)
        sin_theta = np.sin(theta_grid)
        cos_theta = np.cos(theta_grid)
        diffuse_denominator = 0.5 * np.sin(theta_limit_rad) ** 2
        tau_theta = _partition_tau_theta(
            freqs, omega, acoustic_wavenumber, theta_grid, leaves, cavities, ambient
        )
        if size_type == "infinite":
            numerator = np.trapezoid(
                tau_theta * sin_theta[None, :] * cos_theta[None, :], theta_grid, axis=1
            )
            power_tau = numerator / diffuse_denominator
        else:
            power_tau = np.empty(freqs.size, dtype=np.float64)
            for index, ka_i in enumerate(acoustic_wavenumber):
                sigma_theta = _finite_sigma_theta(
                    float(ka_i), sin_theta, window_length, radial_samples
                )
                numerator = np.trapezoid(
                    tau_theta[index] * sigma_theta * sin_theta * np.square(cos_theta),
                    theta_grid,
                )
                power_tau[index] = float(numerator / diffuse_denominator)

    power_tau = np.maximum(np.real(power_tau), 1e-24)
    pressure_frf = np.sqrt(power_tau)
    leaf_descr = "+".join("meta" if leaf.resonator is not None else "bare" for leaf in leaves)
    partition_descr = "single-leaf" if len(leaves) == 1 else f"{len(leaves)}-leaf"
    label = f"{partition_descr} ({leaf_descr}) {size_type} {incidence} analytical filter"
    return AnalyticalFilterResult(
        freqs_hz=freqs,
        pressure_frf=pressure_frf,
        power_tau=power_tau,
        stl_db=transmission_loss_from_power(power_tau),
        label=label,
    )


def double_wall_pressure_filter(
    freqs_hz: np.ndarray | list[float],
    outer_leaf: PartitionLeaf,
    inner_leaf: PartitionLeaf,
    cavity: AirCavity,
    **kwargs,
) -> AnalyticalFilterResult:
    """Convenience wrapper for a mass-air-mass double-wall partition."""
    return partition_pressure_filter(
        freqs_hz,
        (outer_leaf, inner_leaf),
        (cavity,),
        **kwargs,
    )
