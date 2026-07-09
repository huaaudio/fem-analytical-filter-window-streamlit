from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .params import expand_grid


@dataclass(frozen=True, slots=True)
class NastranConfig:
    parallel: int
    dmparallel: int
    max_concurrency: int | None
    executable: str = "nastran"
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FemPrerunConfig:
    solver_id: str
    base_dat: Path
    grid: dict
    nastran: NastranConfig
    work_dir: Path

    def variants(self):
        return expand_grid(self.grid)


def load_config(path) -> FemPrerunConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    nastran_payload = payload.get("nastran", {})
    nastran = NastranConfig(
        parallel=int(nastran_payload.get("parallel", 8)),
        dmparallel=int(nastran_payload.get("dmparallel", 4)),
        max_concurrency=(None if nastran_payload.get("max_concurrency") is None
                         else int(nastran_payload["max_concurrency"])),
        executable=str(nastran_payload.get("executable", "nastran")),
        arguments=tuple(str(arg) for arg in nastran_payload.get("arguments", [])),
    )
    return FemPrerunConfig(
        solver_id=str(payload.get("solver_id", "fem_sol108_diffuse")),
        base_dat=resolve(payload["base_dat"]),
        grid=payload["grid"],
        nastran=nastran,
        work_dir=resolve(payload.get("work_dir", ".cache/fem_prerun")),
    )
