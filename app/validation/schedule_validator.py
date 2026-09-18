"""Independent, deterministic replay of the final schedule (Problem Statement Section 9 & 11).

This runs completely separately from the optimizer's own bookkeeping. If the
optimizer produced an invalid result, this validator catches it and the
service treats it as an internal failure rather than returning bad output.
"""

from dataclasses import dataclass

from app.directives.normalizer import NormalizedConstraints
from app.optimizer.solver import HourResult
from app.schemas import BatterySpec, HourEntry

TOLERANCE = 0.01


@dataclass
class ValidationOutcome:
    valid: bool
    errors: list[str]


def validate_schedule(
    hourly: list[HourResult],
    hours: list[HourEntry],
    battery: BatterySpec,
    constraints: NormalizedConstraints,
) -> ValidationOutcome:
    errors: list[str] = []

    if len(hourly) != 24:
        return ValidationOutcome(False, ["hourly_plan must contain exactly 24 entries"])

    seen_hours = set()
    by_hour = {}
    for entry in hourly:
        if entry.hour in seen_hours:
            errors.append(f"duplicate hour {entry.hour} in hourly_plan")
        seen_hours.add(entry.hour)
        by_hour[entry.hour] = entry
    if seen_hours != set(range(24)):
        errors.append("hourly_plan must cover hours 0..23 exactly once")
        return ValidationOutcome(False, errors)

    demand_by_hour = {h.hour: h.demand_kwh for h in hours}
    solar_by_hour = {h.hour: h.solar_kwh for h in hours}

    prev_energy = battery.initial_energy_kwh

    for h in range(24):
        entry = by_hour[h]

        if entry.grid_kwh < -TOLERANCE:
            errors.append(f"hour {h}: negative grid_kwh")

        effective_solar = max(0.0, solar_by_hour[h] * constraints.solar_factor[h])
        if entry.solar_used_kwh < -TOLERANCE or entry.solar_used_kwh > effective_solar + TOLERANCE:
            errors.append(f"hour {h}: solar_used_kwh exceeds effective solar")

        if entry.battery_action not in ("charge", "discharge", "idle"):
            errors.append(f"hour {h}: invalid battery_action")
        if entry.battery_action == "idle" and abs(entry.battery_kwh) > TOLERANCE:
            errors.append(f"hour {h}: idle hour must have battery_kwh = 0")
        if entry.battery_kwh < -TOLERANCE:
            errors.append(f"hour {h}: negative battery_kwh")

        if entry.battery_action == "charge":
            max_allowed = 0.0 if constraints.no_charge[h] else battery.max_charge_kwh_per_hour
            if entry.battery_kwh > max_allowed + TOLERANCE:
                errors.append(f"hour {h}: charge exceeds max_charge_kwh_per_hour or no_charge_window")
            expected_energy = prev_energy + entry.battery_kwh
        elif entry.battery_action == "discharge":
            max_allowed = 0.0 if constraints.no_discharge[h] else battery.max_discharge_kwh_per_hour
            if entry.battery_kwh > max_allowed + TOLERANCE:
                errors.append(f"hour {h}: discharge exceeds max_discharge_kwh_per_hour or no_discharge_window")
            expected_energy = prev_energy - entry.battery_kwh
        else:
            expected_energy = prev_energy

        if abs(entry.battery_energy_after_kwh - expected_energy) > TOLERANCE:
            errors.append(f"hour {h}: battery_energy_after_kwh inconsistent with battery action")

        reserve = max(0.0, constraints.min_reserve[h])
        if entry.battery_energy_after_kwh < reserve - TOLERANCE:
            errors.append(f"hour {h}: battery energy below active minimum reserve")
        if entry.battery_energy_after_kwh > battery.capacity_kwh + TOLERANCE:
            errors.append(f"hour {h}: battery energy exceeds capacity")

        charge_amt = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge_amt = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
        balance_lhs = entry.grid_kwh + entry.solar_used_kwh + discharge_amt
        balance_rhs = demand_by_hour[h] + charge_amt
        if abs(balance_lhs - balance_rhs) > TOLERANCE:
            errors.append(f"hour {h}: energy balance violated")

        grid_cap = constraints.max_grid[h]
        if grid_cap != float("inf") and entry.grid_kwh > grid_cap + TOLERANCE:
            errors.append(f"hour {h}: grid_kwh exceeds max_grid_window cap")

        prev_energy = entry.battery_energy_after_kwh

    if abs(prev_energy - battery.initial_energy_kwh) > TOLERANCE:
        errors.append("final battery energy does not equal initial_energy_kwh")

    return ValidationOutcome(valid=len(errors) == 0, errors=errors)
