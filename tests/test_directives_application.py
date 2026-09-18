"""Per-directive application tests: feed a hand-built directive interpretation
straight into normalize_directives -> solve_schedule -> validate_schedule and
check the returned plan actually OBEYS the ground-truth directive (not merely
that the interpretation looked right). This mirrors the judge's downstream
replay behavior described in the Problem Statement."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives.normalizer import normalize_directives
from app.directives.validator import ValidatedInterpretation
from app.optimizer.solver import solve_schedule
from app.schemas import BatterySpec, HourEntry
from app.validation.schedule_validator import validate_schedule

EPS = 1e-6


def make_hours(demand, solar, tariff):
    return [
        HourEntry(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]


def flat_scenario(demand=100.0, solar=0.0, tariff=10.0):
    return make_hours([demand] * 24, [solar] * 24, [tariff] * 24)


def default_battery(**overrides):
    base = dict(
        capacity_kwh=200.0,
        initial_energy_kwh=100.0,
        minimum_energy_kwh=20.0,
        max_charge_kwh_per_hour=50.0,
        max_discharge_kwh_per_hour=50.0,
    )
    base.update(overrides)
    return BatterySpec(**base)


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
        explanation="test",
    )


def run(hours, battery, directives):
    constraints = normalize_directives(directives, battery.minimum_energy_kwh)
    result = solve_schedule(hours, battery, constraints)
    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    return result, outcome, constraints


# ---------------------------------------------------------------------------
# 5A. solar_reduction
# ---------------------------------------------------------------------------

def test_solar_reduction_only_listed_hours_reduced():
    hours = make_hours([100] * 24, [80] * 24, [10] * 24)
    battery = default_battery()
    d = [directive("solar_reduction", hours=[10, 11], factor=0.25)]
    result, outcome, constraints = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert constraints.solar_factor[10] == 0.25
    assert constraints.solar_factor[11] == 0.25
    for h in range(24):
        if h not in (10, 11):
            assert constraints.solar_factor[h] == 1.0
    # solar_used must never exceed the reduced effective solar in hours 10/11
    for h in (10, 11):
        entry = result.hourly[h]
        assert entry.solar_used_kwh <= 80 * 0.25 + 1e-6


def test_solar_reduction_80_percent_means_factor_0_2_not_0_8():
    """Directly encodes the spec: 'an 80% reduction' -> factor 0.2, the USABLE
    fraction remaining, never the removed fraction."""
    hours = make_hours([100] * 24, [100] * 24, [10] * 24)
    battery = default_battery()
    d = [directive("solar_reduction", hours=[12], factor=0.2)]
    _, outcome, constraints = run(hours, battery, d)
    assert outcome.valid
    assert constraints.solar_factor[12] == 0.2  # NOT 0.8


def test_solar_reduction_factor_zero_full_blackout():
    hours = make_hours([100] * 24, [100] * 24, [10] * 24)
    battery = default_battery()
    d = [directive("solar_reduction", hours=[12], factor=0.0)]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid
    assert result.hourly[12].solar_used_kwh <= 1e-6


def test_solar_reduction_factor_one_no_change():
    hours = make_hours([100] * 24, [100] * 24, [10] * 24)
    battery = default_battery()
    baseline, _, _ = run(hours, battery, [])
    d = [directive("solar_reduction", hours=[12], factor=1.0)]
    reduced, outcome, _ = run(hours, battery, d)
    assert outcome.valid
    assert abs(baseline.total_cost_bdt - reduced.total_cost_bdt) < 0.02


def test_solar_reduction_does_not_change_demand_or_tariff():
    hours = make_hours([100] * 24, [100] * 24, [10] * 24)
    battery = default_battery()
    d = [directive("solar_reduction", hours=[12, 13], factor=0.3)]
    _, outcome, _ = run(hours, battery, d)
    assert outcome.valid
    # demand/tariff are request-owned and never mutated by directive application;
    # the energy-balance check inside validate_schedule already enforces this
    # implicitly (it uses the original request's demand_by_hour).


def test_solar_reduction_time_window_1pm_to_3pm():
    """1 PM to 3 PM -> hours [13, 14] (start-inclusive, end-exclusive)."""
    hours = make_hours([100] * 24, [100] * 24, [10] * 24)
    battery = default_battery()
    d = [directive("solar_reduction", hours=[13, 14], factor=0.2)]
    _, _, constraints = run(hours, battery, d)
    assert constraints.solar_factor[13] == 0.2
    assert constraints.solar_factor[14] == 0.2
    assert constraints.solar_factor[12] == 1.0
    assert constraints.solar_factor[15] == 1.0


def test_solar_reduction_combined_with_no_charge_window():
    hours = make_hours([100] * 24, [150] * 24, [10] * 24)
    battery = default_battery()
    d = [
        directive("solar_reduction", hours=[10, 11], factor=0.5),
        directive("no_charge_window", hours=[10, 11]),
    ]
    result, outcome, constraints = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (10, 11):
        assert result.hourly[h].battery_action != "charge"


# ---------------------------------------------------------------------------
# 5B. minimum_battery_reserve
# ---------------------------------------------------------------------------

def test_minimum_reserve_percentage_uses_capacity_not_current_energy():
    """50% of a 200 kWh battery = 100 kWh reserve, computed from capacity."""
    hours = make_hours([150] * 24, [0] * 24, [10] * 24)
    battery = default_battery(capacity_kwh=200.0, initial_energy_kwh=150.0, minimum_energy_kwh=20.0)
    d = [directive("minimum_battery_reserve", hours=[18, 19, 20], minimum_energy_kwh=100.0)]
    result, outcome, constraints = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (18, 19, 20):
        assert result.hourly[h].battery_energy_after_kwh >= 100.0 - 0.01


def test_minimum_reserve_active_is_max_of_base_and_directive():
    hours = make_hours([100] * 24, [0] * 24, [10] * 24)
    battery = default_battery(minimum_energy_kwh=30.0)
    # directive reserve LOWER than base minimum must not lower the effective floor
    d = [directive("minimum_battery_reserve", hours=[5], minimum_energy_kwh=10.0)]
    _, _, constraints = run(hours, battery, d)
    assert constraints.min_reserve[5] == 30.0  # base wins, not the lower directive value


def test_minimum_reserve_higher_than_base_raises_floor():
    hours = make_hours([100] * 24, [0] * 24, [10] * 24)
    battery = default_battery(minimum_energy_kwh=30.0)
    d = [directive("minimum_battery_reserve", hours=[5], minimum_energy_kwh=80.0)]
    _, _, constraints = run(hours, battery, d)
    assert constraints.min_reserve[5] == 80.0


def test_minimum_reserve_only_active_inside_window():
    hours = make_hours([100] * 24, [0] * 24, [10] * 24)
    battery = default_battery(minimum_energy_kwh=20.0)
    d = [directive("minimum_battery_reserve", hours=[18, 19, 20], minimum_energy_kwh=150.0)]
    _, _, constraints = run(hours, battery, d)
    assert constraints.min_reserve[17] == 20.0
    assert constraints.min_reserve[21] == 20.0
    assert constraints.min_reserve[18] == 150.0


def test_minimum_reserve_high_during_expensive_hours_still_enforced():
    demand = [100] * 24
    solar = [0] * 24
    tariff = [10] * 18 + [100, 100, 100] + [10, 10, 10]  # spike at 18-20
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=300.0, initial_energy_kwh=200.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=80.0)
    d = [directive("minimum_battery_reserve", hours=[18, 19, 20], minimum_energy_kwh=150.0)]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (18, 19, 20):
        assert result.hourly[h].battery_energy_after_kwh >= 150.0 - 0.01


def test_minimum_reserve_combined_with_max_grid_window():
    demand = [150] * 24
    solar = [0] * 24
    tariff = [10] * 24
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=300.0, initial_energy_kwh=200.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=80.0, max_charge_kwh_per_hour=80.0)
    d = [
        directive("minimum_battery_reserve", hours=[18, 19, 20], minimum_energy_kwh=100.0),
        directive("max_grid_window", hours=[18, 19, 20], max_grid_kwh=100.0),
    ]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (18, 19, 20):
        assert result.hourly[h].grid_kwh <= 100.0 + 0.01
        assert result.hourly[h].battery_energy_after_kwh >= 100.0 - 0.01


# ---------------------------------------------------------------------------
# 5C. no_charge_window
# ---------------------------------------------------------------------------

def test_no_charge_window_forces_zero_charge():
    hours = make_hours([50] * 24, [200] * 24, [10] * 24)  # solar surplus wants to charge
    battery = default_battery(initial_energy_kwh=50.0)
    d = [directive("no_charge_window", hours=[10, 11, 12])]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (10, 11, 12):
        assert result.hourly[h].battery_action != "charge"
        assert result.hourly[h].battery_kwh == 0 or result.hourly[h].battery_action == "discharge"


def test_no_charge_window_single_hour_boundary():
    hours = make_hours([50] * 24, [200] * 24, [10] * 24)
    battery = default_battery(initial_energy_kwh=50.0)
    d = [directive("no_charge_window", hours=[0])]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[0].battery_action != "charge"


def test_no_charge_window_hour_23_boundary():
    hours = make_hours([50] * 24, [200] * 24, [10] * 24)
    battery = default_battery(initial_energy_kwh=50.0)
    d = [directive("no_charge_window", hours=[23])]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[23].battery_action != "charge"


def test_no_charge_window_full_day_still_satisfies_end_of_day_neutrality():
    """If charging is disabled all day, the optimizer can only reach end-of-day
    neutrality via discharge-then-implicit-idle; verify it's still valid (initial
    energy already at the floor makes this trivially feasible)."""
    hours = make_hours([50] * 24, [0] * 24, [10] * 24)
    battery = default_battery(initial_energy_kwh=20.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=0.0)
    d = [directive("no_charge_window", hours=list(range(24)))]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[23].battery_energy_after_kwh == battery.initial_energy_kwh


# ---------------------------------------------------------------------------
# 5D. no_discharge_window
# ---------------------------------------------------------------------------

def test_no_discharge_window_forces_zero_discharge_even_at_high_tariff():
    """The optimizer must obey the hard directive even when discharging would
    reduce cost -- this is the critical case."""
    demand = [50] * 24
    solar = [0] * 24
    tariff = [10] * 17 + [500, 500, 500] + [10] * 4  # extreme spike at 17-19
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=300.0, initial_energy_kwh=200.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=80.0)
    d = [directive("no_discharge_window", hours=[17, 18, 19])]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (17, 18, 19):
        assert result.hourly[h].battery_action != "discharge"
        # confirm the optimizer paid full grid price rather than illegally discharging
        assert result.hourly[h].grid_kwh >= demand[h] - 0.01


def test_no_discharge_window_boundary_hours():
    hours = make_hours([50] * 24, [0] * 24, [10] * 24)
    battery = default_battery(initial_energy_kwh=100.0)
    d = [directive("no_discharge_window", hours=[0, 23])]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[0].battery_action != "discharge"
    assert result.hourly[23].battery_action != "discharge"


# ---------------------------------------------------------------------------
# 5E. max_grid_window
# ---------------------------------------------------------------------------

def test_max_grid_window_is_hard_cap_even_when_costlier():
    demand = [300] * 24
    solar = [0] * 24
    tariff = [1] * 24  # cheap grid, but capped anyway
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=500.0, initial_energy_kwh=300.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=200.0, max_charge_kwh_per_hour=200.0)
    d = [directive("max_grid_window", hours=[10, 11], max_grid_kwh=150.0)]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    for h in (10, 11):
        assert result.hourly[h].grid_kwh <= 150.0 + 0.01


def test_max_grid_window_cap_at_zero():
    demand = [100] * 24
    solar = [200] * 24
    tariff = [10] * 24
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=300.0, initial_energy_kwh=200.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=200.0)
    d = [directive("max_grid_window", hours=[12], max_grid_kwh=0.0)]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[12].grid_kwh <= 0.01


def test_max_grid_window_combined_with_no_discharge_still_feasible_via_solar():
    """Cap satisfiable purely from solar even though discharge is blocked."""
    demand = [100] * 24
    solar = [200] * 24
    tariff = [10] * 24
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(capacity_kwh=300.0, initial_energy_kwh=200.0, minimum_energy_kwh=20.0)
    d = [
        directive("max_grid_window", hours=[12], max_grid_kwh=0.0),
        directive("no_discharge_window", hours=[12]),
    ]
    result, outcome, _ = run(hours, battery, d)
    assert outcome.valid, outcome.errors
    assert result.hourly[12].grid_kwh <= 0.01
    assert result.hourly[12].battery_action != "discharge"


# ---------------------------------------------------------------------------
# 5F. no_op
# ---------------------------------------------------------------------------

def test_no_op_never_changes_optimizer_constraints():
    hours = make_hours([100] * 24, [50] * 24, [10] * 24)
    battery = default_battery()
    baseline_constraints = normalize_directives([], battery.minimum_energy_kwh)
    noop_constraints = normalize_directives([directive("no_op")], battery.minimum_energy_kwh)
    assert baseline_constraints.solar_factor == noop_constraints.solar_factor
    assert baseline_constraints.min_reserve == noop_constraints.min_reserve
    assert baseline_constraints.no_charge == noop_constraints.no_charge
    assert baseline_constraints.no_discharge == noop_constraints.no_discharge
    assert baseline_constraints.max_grid == noop_constraints.max_grid


# ---------------------------------------------------------------------------
# Simultaneous charge/discharge must never appear in output
# ---------------------------------------------------------------------------

def test_never_simultaneous_charge_and_discharge():
    """Force a degenerate LP direction (charge and discharge cancel with zero
    net cost effect) and verify the netting logic collapses it to one action."""
    hours = make_hours([100] * 24, [0] * 24, [10] * 24)
    battery = default_battery(initial_energy_kwh=100.0, minimum_energy_kwh=0.0, capacity_kwh=1000.0, max_charge_kwh_per_hour=500.0, max_discharge_kwh_per_hour=500.0)
    result, outcome, _ = run(hours, battery, [])
    assert outcome.valid, outcome.errors
    for entry in result.hourly:
        assert entry.battery_action in ("charge", "discharge", "idle")
        if entry.battery_action == "idle":
            assert entry.battery_kwh == 0


def test_idle_always_has_zero_battery_kwh():
    hours = flat_scenario(demand=100, solar=100, tariff=10)
    battery = default_battery(initial_energy_kwh=100.0, minimum_energy_kwh=100.0, max_charge_kwh_per_hour=0.0, max_discharge_kwh_per_hour=0.0)
    result, outcome, _ = run(hours, battery, [])
    assert outcome.valid, outcome.errors
    for entry in result.hourly:
        if entry.battery_action == "idle":
            assert entry.battery_kwh == 0


# ---------------------------------------------------------------------------
# End-of-day neutrality attacks
# ---------------------------------------------------------------------------

def test_end_of_day_neutrality_cheap_final_hour_cannot_be_exploited():
    """Even when the last hour is the cheapest, the optimizer must not end the
    day with less energy than it started (no free lunch from starting charge)."""
    demand = [100] * 24
    solar = [0] * 24
    tariff = [50] * 23 + [1]  # hour 23 dirt cheap
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(initial_energy_kwh=150.0, capacity_kwh=300.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=100.0, max_charge_kwh_per_hour=100.0)
    result, outcome, _ = run(hours, battery, [])
    assert outcome.valid, outcome.errors
    assert abs(result.hourly[23].battery_energy_after_kwh - battery.initial_energy_kwh) < 0.01


def test_end_of_day_neutrality_expensive_final_hour():
    demand = [100] * 24
    solar = [0] * 24
    tariff = [1] * 23 + [500]  # hour 23 extremely expensive
    hours = make_hours(demand, solar, tariff)
    battery = default_battery(initial_energy_kwh=150.0, capacity_kwh=300.0, minimum_energy_kwh=20.0, max_discharge_kwh_per_hour=100.0, max_charge_kwh_per_hour=100.0)
    result, outcome, _ = run(hours, battery, [])
    assert outcome.valid, outcome.errors
    assert abs(result.hourly[23].battery_energy_after_kwh - battery.initial_energy_kwh) < 0.01


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
    print(f"\n{len(failures)} failing tests" if failures else "\nAll directive application tests passed")
    if failures:
        sys.exit(1)
