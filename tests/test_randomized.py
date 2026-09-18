"""Property-based / randomized testing. Generates many random feasible 24-hour
scenarios (random demand, solar, tariff, battery specs, and directive
combinations) and independently validates every returned schedule. Also
includes boundary-heavy deterministic cases (values exactly at limits)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives.normalizer import normalize_directives
from app.directives.validator import ValidatedInterpretation
from app.optimizer.solver import solve_schedule
from app.schemas import BatterySpec, HourEntry
from app.validation.schedule_validator import validate_schedule

N_RANDOM_CASES = 60


def directive(directive_type, hours=None, **adj_fields):
    adjustment = None
    if directive_type != "no_op":
        adjustment = {"hours": hours}
        adjustment.update(adj_fields)
    return ValidatedInterpretation(
        note_index=0,
        applies=(directive_type != "no_op"),
        directive_type=directive_type,
        structured_adjustment=adjustment,
        explanation="random",
    )


def random_scenario(rng: random.Random):
    demand = [round(rng.uniform(20, 250), 2) for _ in range(24)]
    solar = [round(rng.uniform(0, 200) if 6 <= h <= 18 else 0.0, 2) for h in range(24)]
    tariff = [round(rng.uniform(1, 40), 2) for _ in range(24)]
    hours = [
        HourEntry(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]

    capacity = round(rng.uniform(100, 500), 2)
    minimum = round(rng.uniform(0, capacity * 0.3), 2)
    initial = round(rng.uniform(minimum, capacity), 2)
    max_charge = round(rng.uniform(10, capacity * 0.5), 2)
    max_discharge = round(rng.uniform(10, capacity * 0.5), 2)

    battery = BatterySpec(
        capacity_kwh=capacity,
        initial_energy_kwh=initial,
        minimum_energy_kwh=minimum,
        max_charge_kwh_per_hour=max_charge,
        max_discharge_kwh_per_hour=max_discharge,
    )
    return hours, battery


def random_compatible_directives(rng: random.Random, battery: BatterySpec):
    """Generate 0-2 directives that are compatible with each other (no
    contradictory hard constraints), mirroring 'organizer valid scoring
    scenarios are feasible' from the Problem Statement."""
    directives = []
    choices = rng.sample(
        ["solar_reduction", "minimum_battery_reserve", "no_charge_window", "max_grid_window", "none"],
        k=2,
    )
    used_hours = set()
    for choice in choices:
        if choice == "none":
            continue
        start = rng.randint(0, 20)
        end = min(24, start + rng.randint(1, 3))
        hrs = list(range(start, end))
        if any(h in used_hours for h in hrs):
            continue  # avoid overlapping hard constraints from different notes
        used_hours.update(hrs)

        if choice == "solar_reduction":
            directives.append(directive("solar_reduction", hours=hrs, factor=round(rng.uniform(0, 1), 2)))
        elif choice == "minimum_battery_reserve":
            reserve = round(rng.uniform(0, battery.capacity_kwh * 0.6), 2)
            directives.append(directive("minimum_battery_reserve", hours=hrs, minimum_energy_kwh=reserve))
        elif choice == "no_charge_window":
            directives.append(directive("no_charge_window", hours=hrs))
        elif choice == "max_grid_window":
            cap = round(rng.uniform(50, 400), 2)
            directives.append(directive("max_grid_window", hours=hrs, max_grid_kwh=cap))
    return directives


def test_random_feasible_scenarios_always_produce_valid_schedules():
    rng = random.Random(1234)
    failures = []
    for i in range(N_RANDOM_CASES):
        hours, battery = random_scenario(rng)
        directives = random_compatible_directives(rng, battery)
        constraints = normalize_directives(directives, battery.minimum_energy_kwh)
        try:
            result = solve_schedule(hours, battery, constraints)
        except Exception as exc:
            failures.append(f"case {i}: solver raised {type(exc).__name__}: {exc}")
            continue
        outcome = validate_schedule(result.hourly, hours, battery, constraints)
        if not outcome.valid:
            failures.append(f"case {i}: {outcome.errors}")
    assert not failures, "\n".join(failures)


def test_boundary_capacity_equals_initial():
    hours = [HourEntry(hour=h, demand_kwh=100, solar_kwh=0, tariff_bdt_per_kwh=10) for h in range(24)]
    battery = BatterySpec(capacity_kwh=200, initial_energy_kwh=200, minimum_energy_kwh=0, max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    constraints = normalize_directives([], battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    assert outcome.valid, outcome.errors


def test_boundary_reserve_equals_capacity():
    """Battery must stay pinned at full capacity for the whole day."""
    hours = [HourEntry(hour=h, demand_kwh=100, solar_kwh=0, tariff_bdt_per_kwh=10) for h in range(24)]
    battery = BatterySpec(capacity_kwh=200, initial_energy_kwh=200, minimum_energy_kwh=200, max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    constraints = normalize_directives([], battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    assert outcome.valid, outcome.errors
    for entry in result.hourly:
        assert entry.battery_action == "idle"


def test_boundary_zero_charge_and_discharge_rate():
    """Battery is frozen: it can neither charge nor discharge all day."""
    hours = [HourEntry(hour=h, demand_kwh=100, solar_kwh=0, tariff_bdt_per_kwh=10) for h in range(24)]
    battery = BatterySpec(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=50, max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    constraints = normalize_directives([], battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    assert outcome.valid, outcome.errors
    for entry in result.hourly:
        assert entry.battery_action == "idle"
        assert entry.grid_kwh == 100  # must buy full demand from grid every hour


def test_boundary_solar_exactly_equals_demand():
    hours = [HourEntry(hour=h, demand_kwh=100, solar_kwh=100, tariff_bdt_per_kwh=10) for h in range(24)]
    battery = BatterySpec(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20, max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    constraints = normalize_directives([], battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    assert outcome.valid, outcome.errors
    assert result.total_cost_bdt < 1.0  # near-zero grid needed


def test_boundary_solar_far_exceeds_demand():
    hours = [HourEntry(hour=h, demand_kwh=50, solar_kwh=1000, tariff_bdt_per_kwh=10) for h in range(24)]
    battery = BatterySpec(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20, max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    constraints = normalize_directives([], battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    assert outcome.valid, outcome.errors
    # unused solar must be curtailed, never exported (no negative grid anywhere)
    for entry in result.hourly:
        assert entry.grid_kwh >= 0


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failures.append(name)
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failures.append(name)
    print(f"\n{len(failures)} failing tests" if failures else "\nAll randomized tests passed")
    if failures:
        sys.exit(1)
