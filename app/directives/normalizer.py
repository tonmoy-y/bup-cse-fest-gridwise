"""Turns validated directive interpretations into per-hour optimizer constraints."""

from dataclasses import dataclass, field

from app.directives.validator import ValidatedInterpretation


@dataclass
class NormalizedConstraints:
    solar_factor: list[float] = field(default_factory=lambda: [1.0] * 24)
    min_reserve: list[float] = field(default_factory=lambda: [0.0] * 24)
    no_charge: list[bool] = field(default_factory=lambda: [False] * 24)
    no_discharge: list[bool] = field(default_factory=lambda: [False] * 24)
    max_grid: list[float] = field(default_factory=lambda: [float("inf")] * 24)


def normalize_directives(
    interpretations: list[ValidatedInterpretation], base_minimum_energy_kwh: float
) -> NormalizedConstraints:
    constraints = NormalizedConstraints()
    constraints.min_reserve = [base_minimum_energy_kwh] * 24

    for item in interpretations:
        if not item.applies or item.structured_adjustment is None:
            continue
        adj = item.structured_adjustment
        hours = adj.get("hours", [])

        if item.directive_type == "solar_reduction":
            factor = adj["factor"]
            for h in hours:
                constraints.solar_factor[h] *= factor

        elif item.directive_type == "minimum_battery_reserve":
            min_energy = adj["minimum_energy_kwh"]
            for h in hours:
                constraints.min_reserve[h] = max(constraints.min_reserve[h], min_energy)

        elif item.directive_type == "no_charge_window":
            for h in hours:
                constraints.no_charge[h] = True

        elif item.directive_type == "no_discharge_window":
            for h in hours:
                constraints.no_discharge[h] = True

        elif item.directive_type == "max_grid_window":
            cap = adj["max_grid_kwh"]
            for h in hours:
                constraints.max_grid[h] = min(constraints.max_grid[h], cap)

    return constraints
