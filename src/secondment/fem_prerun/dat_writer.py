from __future__ import annotations

import hashlib
from pathlib import Path

from .nastran_real import format_nastran_real8
from .params import VariantParams

MASS_SLICE = slice(32, 40)   # CONM2 mass, columns 33-40
K1_SLICE = slice(24, 32)     # PBUSH K1, columns 25-32
GE_SLICE = slice(24, 32)     # PBUSH GE continuation, columns 25-32
FLAG_SLICE = slice(16, 24)   # columns 17-24 (holds the "K"/"GE" flag)

EXPECTED_CONM2 = 48


def hash_base_dat(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split_eol(line: str) -> tuple[str, str]:
    for eol in ("\r\n", "\n", "\r"):
        if line.endswith(eol):
            return line[: -len(eol)], eol
    return line, ""


def _replace_slice(content: str, where: slice, value: str) -> str:
    field = format_nastran_real8(value)
    if len(content) < where.stop:
        raise ValueError(f"Line too short to hold field {where}: {content!r}")
    return content[: where.start] + field + content[where.stop :]


def render_variant_dat(base_text: str, variant: VariantParams) -> str:
    mass_field = variant.m_res
    k_field = variant.k_res
    ge_field = variant.ge

    conm2_count = 0
    pbush_count = 0
    ge_count = 0
    out: list[str] = []

    for raw_line in base_text.splitlines(keepends=True):
        content, eol = _split_eol(raw_line)
        if content.startswith("CONM2"):
            content = _replace_slice(content, MASS_SLICE, mass_field)
            conm2_count += 1
        elif content.startswith("PBUSH"):
            content = _replace_slice(content, K1_SLICE, k_field)
            pbush_count += 1
        elif content.startswith("+") and content[FLAG_SLICE].strip() == "GE":
            content = _replace_slice(content, GE_SLICE, ge_field)
            ge_count += 1
        out.append(content + eol)

    if conm2_count != EXPECTED_CONM2:
        raise ValueError(f"Expected {EXPECTED_CONM2} CONM2 rewrites, got {conm2_count}")
    if pbush_count != 1:
        raise ValueError(f"Expected 1 PBUSH rewrite, got {pbush_count}")
    if ge_count != 1:
        raise ValueError(f"Expected 1 GE rewrite, got {ge_count}")

    return "".join(out)
