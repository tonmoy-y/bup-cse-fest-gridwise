"""Deterministic guardrails for LLM-produced directive interpretations.

Raw LLM output is never trusted directly. Every entry is checked against the
exact rules in the Problem Statement (Section 08) before it can reach the
optimizer. Any entry that fails validation is safely downgraded to no_op
instead of being silently invented or crashing the service.
"""

from dataclasses import dataclass

ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

REQUIRED_FIELDS = {
    "solar_reduction": ("hours", "factor"),
    "minimum_battery_reserve": ("hours", "minimum_energy_kwh"),
    "no_charge_window": ("hours",),
    "no_discharge_window": ("hours",),
    "max_grid_window": ("hours", "max_grid_kwh"),
    "no_op": (),
}


@dataclass
class ValidatedInterpretation:
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: dict | None
    explanation: str
    was_downgraded: bool = False
    """True when this entry required a safe guardrail fallback because the raw
    LLM output for this note was missing, malformed, or semantically invalid.
    Used by the failover manager to decide whether a provider's result counts
    as a clean success or whether another candidate should be tried."""


def _safe_no_op(note_index: int, explanation: str) -> ValidatedInterpretation:
    return ValidatedInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=explanation,
        was_downgraded=True,
    )


def _validate_hours(hours) -> list[int] | None:
    if not isinstance(hours, list) or len(hours) == 0:
        return None
    if any(isinstance(h, bool) for h in hours):
        return None
    int_hours = []
    for h in hours:
        if isinstance(h, int):
            int_hours.append(h)
        elif isinstance(h, float):
            if not h.is_integer():
                return None
            int_hours.append(int(h))
        elif isinstance(h, str):
            try:
                int_hours.append(int(h.strip()))
            except ValueError:
                return None
        else:
            return None
    if any(h < 0 or h > 23 for h in int_hours):
        return None
    if len(set(int_hours)) != len(int_hours):
        return None
    if int_hours != sorted(int_hours):
        return None
    return int_hours


def _validate_number(value, minimum=None, maximum=None) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%"))
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    val = float(value)
    if val != val or val in (float("inf"), float("-inf")):
        return None
    if minimum is not None and val < minimum:
        return None
    if maximum is not None and val > maximum:
        return None
    return val


def validate_interpretations(
    raw_interpretations: list[dict],
    operator_notes: list[str],
    battery_capacity_kwh: float | None = None,
) -> list[ValidatedInterpretation]:
    """Validate raw LLM entries against operator_notes, returning one safe entry per note.

    When battery_capacity_kwh is given, a minimum_battery_reserve above the
    battery capacity is rejected (Problem Statement Section 08).
    """
    n = len(operator_notes)
    by_index: dict[int, dict] = {}
    duplicated: set[int] = set()

    for entry in raw_interpretations:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("note_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx < 0 or idx >= n:
            continue
        if idx in by_index:
            duplicated.add(idx)
        else:
            by_index[idx] = entry

    results: list[ValidatedInterpretation] = []
    for i in range(n):
        entry = by_index.get(i)
        if i in duplicated:
            # Conflicting mappings for one note: neither can be trusted.
            results.append(_safe_no_op(i, "Duplicate interpretations for this note; downgraded."))
            continue
        if entry is None:
            results.append(_safe_no_op(i, "No valid interpretation was produced for this note."))
            continue

        directive_type = entry.get("directive_type")
        applies = entry.get("applies")
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Interpretation for this operator note."

        if directive_type not in ALLOWED_DIRECTIVE_TYPES:
            results.append(_safe_no_op(i, "Unsupported or missing directive type."))
            continue

        if directive_type == "no_op":
            results.append(
                ValidatedInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation=explanation,
                )
            )
            continue

        if applies is not True:
            results.append(_safe_no_op(i, "Non-no_op directive missing applies=true; downgraded."))
            continue

        hours = _validate_hours(entry.get("hours"))
        if hours is None:
            results.append(_safe_no_op(i, "Invalid or missing hours array; downgraded to no_op."))
            continue

        if directive_type == "solar_reduction":
            factor = _validate_number(entry.get("factor"), minimum=0.0, maximum=1.0)
            if factor is None:
                results.append(_safe_no_op(i, "Invalid solar_reduction factor; downgraded."))
                continue
            adjustment = {"hours": hours, "factor": factor}

        elif directive_type == "minimum_battery_reserve":
            min_energy = _validate_number(
                entry.get("minimum_energy_kwh"), minimum=0.0, maximum=battery_capacity_kwh
            )
            if min_energy is None:
                results.append(_safe_no_op(i, "Invalid minimum_energy_kwh; downgraded."))
                continue
            adjustment = {"hours": hours, "minimum_energy_kwh": min_energy}

        elif directive_type == "no_charge_window":
            adjustment = {"hours": hours}

        elif directive_type == "no_discharge_window":
            adjustment = {"hours": hours}

        elif directive_type == "max_grid_window":
            max_grid = _validate_number(entry.get("max_grid_kwh"), minimum=0.0)
            if max_grid is None:
                results.append(_safe_no_op(i, "Invalid max_grid_kwh; downgraded."))
                continue
            adjustment = {"hours": hours, "max_grid_kwh": max_grid}

        else:
            results.append(_safe_no_op(i, "Unhandled directive type; downgraded."))
            continue

        results.append(
            ValidatedInterpretation(
                note_index=i,
                applies=True,
                directive_type=directive_type,
                structured_adjustment=adjustment,
                explanation=explanation,
            )
        )

    return results
