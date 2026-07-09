from __future__ import annotations

import math

WIDTH = 8


def format_nastran_real8(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Cannot format non-finite value: {value!r}")
    text = _fixed_point(value) or _exponential(value)
    if len(text) > WIDTH:
        raise ValueError(f"Formatted real {text!r} exceeds {WIDTH} columns for {value!r}")
    return text.rjust(WIDTH)


def _fixed_point(value: float) -> str | None:
    sign = "-" if value < 0 else ""
    body = WIDTH - len(sign)
    magnitude = abs(value)
    for decimals in range(body, -1, -1):
        text = f"{magnitude:.{decimals}f}"
        if "." not in text:
            text += "."
        if len(text) <= body:
            return sign + text
    return None


def _exponential(value: float) -> str:
    sign = "-" if value < 0 else ""
    body = WIDTH - len(sign)
    magnitude = abs(value)
    for sig in range(body, 0, -1):
        formatted = f"{magnitude:.{sig}e}"  # e.g. "1.760000e+07"
        mantissa, exponent = formatted.split("e")
        exp_int = int(exponent)
        exp_str = f"{'+' if exp_int >= 0 else '-'}{abs(exp_int)}"
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        if "." not in mantissa:
            mantissa += "."
        candidate = mantissa + exp_str
        if len(candidate) <= body:
            return sign + candidate
    raise ValueError(f"Cannot format {value!r} into {WIDTH} columns")
