from __future__ import annotations

from dataclasses import dataclass

from secondment.analytical_filter import HostPanel


@dataclass(frozen=True, slots=True)
class Material:
    """Linear-elastic material properties for an analytical partition leaf."""

    density: float
    young_modulus: float
    poisson_ratio: float
    loss_factor: float

    def host_panel(self, thickness_m: float, *, loss_factor: float | None = None) -> HostPanel:
        return HostPanel(
            density=self.density,
            thickness=thickness_m,
            young_modulus=self.young_modulus,
            poisson_ratio=self.poisson_ratio,
            loss_factor=self.loss_factor if loss_factor is None else loss_factor,
        )


# Representative literature values (room-temperature, airborne acoustics use).
MATERIALS: dict[str, Material] = {
    "aluminum": Material(density=2700.0, young_modulus=70e9, poisson_ratio=0.33, loss_factor=0.01),
    "glass": Material(density=2500.0, young_modulus=70e9, poisson_ratio=0.22, loss_factor=0.02),
    "plexiglass": Material(density=1190.0, young_modulus=3.3e9, poisson_ratio=0.37, loss_factor=0.06),
    "steel": Material(density=7850.0, young_modulus=210e9, poisson_ratio=0.30, loss_factor=0.01),
    "gypsum": Material(density=700.0, young_modulus=2.5e9, poisson_ratio=0.30, loss_factor=0.012),
}

# Friendly aliases.
MATERIAL_ALIASES: dict[str, str] = {
    "pmma": "plexiglass",
    "acrylic": "plexiglass",
    "drywall": "gypsum",
    "plasterboard": "gypsum",
    "alu": "aluminum",
    "aluminium": "aluminum",
}


def available_materials() -> tuple[str, ...]:
    return tuple(MATERIALS)


def get_material(name: str) -> Material:
    key = name.strip().lower()
    key = MATERIAL_ALIASES.get(key, key)
    try:
        return MATERIALS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(MATERIALS))
        raise ValueError(f"Unknown material '{name}'. Choose one of: {supported}.") from exc


def host_panel(
    material_name: str,
    thickness_m: float,
    *,
    loss_factor: float | None = None,
) -> HostPanel:
    """Build a :class:`HostPanel` from a named material and thickness."""
    return get_material(material_name).host_panel(thickness_m, loss_factor=loss_factor)
