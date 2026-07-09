from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

RHO = 2794.0
L1 = 0.05
L2 = 0.05
H = 0.002
CELL_FACTOR = RHO * L1 * L2 * H  # 0.01397 kg

MAX_M_RATIO = 0.20
MIN_ETA = 0.001
MAX_ETA = 0.05


def resonator_mass(m_ratio: float) -> float:
    return float(m_ratio) * CELL_FACTOR


def resonator_stiffness(m_ratio: float, f_res_hz: float) -> float:
    return resonator_mass(m_ratio) * (2.0 * math.pi * float(f_res_hz)) ** 2


@dataclass(frozen=True, slots=True)
class VariantParams:
    m_ratio: float
    f_res_hz: float
    eta_res: float

    def __post_init__(self) -> None:
        if not (0.0 < self.m_ratio <= MAX_M_RATIO):
            raise ValueError(f"m_ratio must be in (0, {MAX_M_RATIO}]: {self.m_ratio}")
        if not (self.f_res_hz > 0.0):
            raise ValueError(f"f_res_hz must be > 0: {self.f_res_hz}")
        if not (MIN_ETA <= self.eta_res <= MAX_ETA):
            raise ValueError(f"eta_res must be in [{MIN_ETA}, {MAX_ETA}]: {self.eta_res}")

    @property
    def m_res(self) -> float:
        return resonator_mass(self.m_ratio)

    @property
    def k_res(self) -> float:
        return resonator_stiffness(self.m_ratio, self.f_res_hz)

    @property
    def ge(self) -> float:
        return float(self.eta_res)

    @property
    def variant_id(self) -> str:
        raw = f"m{self.m_ratio:.4f}_f{self.f_res_hz:.2f}_g{self.eta_res:.4f}"
        return raw.replace(".", "p")


def _axis_values(spec) -> list[float]:
    if isinstance(spec, dict):
        start = float(spec["start"])
        stop = float(spec["stop"])
        step = float(spec["step"])
        if step <= 0.0:
            raise ValueError(f"Grid axis step must be > 0: {step}")
        values = []
        value = start
        # inclusive of stop within a small tolerance
        while value <= stop + step * 1e-9:
            values.append(round(value, 9))
            value += step
        return values
    return [float(v) for v in spec]


def expand_grid(grid: dict) -> list[VariantParams]:
    m_ratios = _axis_values(grid["m_ratio"])
    f_values = _axis_values(grid["f_res_hz"])
    etas = _axis_values(grid["eta_res"])
    return [
        VariantParams(m_ratio=m, f_res_hz=f, eta_res=e)
        for m, f, e in product(m_ratios, f_values, etas)
    ]
