"""Deterministic 24-hour cost-minimizing scheduler.

Formulated as a linear program and solved with scipy.optimize.linprog
(HiGHS). Variables per hour h = 0..23:
  grid[h]        >= 0                         grid electricity purchased
  solar_used[h]  in [0, effective_solar[h]]    solar energy consumed
  charge[h]      in [0, max_charge] (0 if no_charge_window)
  discharge[h]   in [0, max_discharge] (0 if no_discharge_window)
  energy[h]      in [reserve[h], capacity]     battery energy after hour h

Constraints:
  energy balance:  grid[h] + solar_used[h] + discharge[h] - charge[h] = demand[h]
  battery update:  energy[h] - energy[h-1] - charge[h] + discharge[h] = 0
                   (energy[-1] := initial_energy_kwh)
  end-of-day:      energy[23] = initial_energy_kwh

Objective: minimize sum(tariff[h] * grid[h])

The LLM never participates in this step; only validated, normalized
constraints reach the solver.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from app.directives.normalizer import NormalizedConstraints
from app.schemas import BatterySpec, HourEntry


class OptimizationError(Exception):
    pass


@dataclass
class HourResult:
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


@dataclass
class OptimizationResult:
    hourly: list[HourResult]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


N_HOURS = 24
# variable layout per hour: [grid, solar_used, charge, discharge, energy]
VARS_PER_HOUR = 5
EPS = 1e-6


def _var_index(h: int, slot: int) -> int:
    return h * VARS_PER_HOUR + slot


def solve_schedule(
    hours: list[HourEntry],
    battery: BatterySpec,
    constraints: NormalizedConstraints,
) -> OptimizationResult:
    hours_sorted = sorted(hours, key=lambda x: x.hour)
    demand = [h.demand_kwh for h in hours_sorted]
    base_solar = [h.solar_kwh for h in hours_sorted]
    tariff = [h.tariff_bdt_per_kwh for h in hours_sorted]

    effective_solar = [
        max(0.0, base_solar[h] * constraints.solar_factor[h]) for h in range(N_HOURS)
    ]

    n_vars = N_HOURS * VARS_PER_HOUR
    c = np.zeros(n_vars)
    bounds: list[tuple[float, float]] = [(0.0, 0.0)] * n_vars

    for h in range(N_HOURS):
        c[_var_index(h, 0)] = tariff[h]

        grid_ub = constraints.max_grid[h]
        bounds[_var_index(h, 0)] = (0.0, grid_ub if grid_ub != float("inf") else None)
        bounds[_var_index(h, 1)] = (0.0, effective_solar[h])

        max_charge = 0.0 if constraints.no_charge[h] else battery.max_charge_kwh_per_hour
        bounds[_var_index(h, 2)] = (0.0, max_charge)

        max_discharge = 0.0 if constraints.no_discharge[h] else battery.max_discharge_kwh_per_hour
        bounds[_var_index(h, 3)] = (0.0, max_discharge)

        reserve = max(0.0, constraints.min_reserve[h])
        bounds[_var_index(h, 4)] = (reserve, battery.capacity_kwh)

    A_eq = []
    b_eq = []

    for h in range(N_HOURS):
        row = np.zeros(n_vars)
        row[_var_index(h, 0)] = 1.0
        row[_var_index(h, 1)] = 1.0
        row[_var_index(h, 3)] = 1.0
        row[_var_index(h, 2)] = -1.0
        A_eq.append(row)
        b_eq.append(demand[h])

    for h in range(N_HOURS):
        row = np.zeros(n_vars)
        row[_var_index(h, 4)] = 1.0
        row[_var_index(h, 2)] = -1.0
        row[_var_index(h, 3)] = 1.0
        if h > 0:
            row[_var_index(h - 1, 4)] = -1.0
            b_eq.append(0.0)
        else:
            b_eq.append(battery.initial_energy_kwh)
        A_eq.append(row)

    row_final = np.zeros(n_vars)
    row_final[_var_index(N_HOURS - 1, 4)] = 1.0
    A_eq.append(row_final)
    b_eq.append(battery.initial_energy_kwh)

    result = linprog(
        c=c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise OptimizationError(f"LP solver failed: {result.message}")

    x = result.x
    hourly: list[HourResult] = []
    prev_energy = battery.initial_energy_kwh
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(N_HOURS):
        grid_kwh = max(0.0, x[_var_index(h, 0)])
        solar_used = max(0.0, x[_var_index(h, 1)])
        charge = max(0.0, x[_var_index(h, 2)])
        discharge = max(0.0, x[_var_index(h, 3)])
        energy_after = x[_var_index(h, 4)]

        net = charge - discharge
        if net > EPS:
            action = "charge"
            battery_kwh = net
        elif net < -EPS:
            action = "discharge"
            battery_kwh = -net
        else:
            action = "idle"
            battery_kwh = 0.0

        # recompute energy_after from the netted action to guarantee internal consistency
        energy_after = prev_energy + (battery_kwh if action == "charge" else 0.0) - (
            battery_kwh if action == "discharge" else 0.0
        )
        prev_energy = energy_after

        total_grid += grid_kwh
        total_cost += grid_kwh * tariff[h]
        peak_grid = max(peak_grid, grid_kwh)

        hourly.append(
            HourResult(
                hour=h,
                grid_kwh=round(grid_kwh, 6),
                solar_used_kwh=round(solar_used, 6),
                battery_action=action,
                battery_kwh=round(battery_kwh, 6),
                battery_energy_after_kwh=round(energy_after, 6),
            )
        )

    return OptimizationResult(
        hourly=hourly,
        total_grid_kwh=round(total_grid, 6),
        total_cost_bdt=round(total_cost, 6),
        peak_grid_kwh=round(peak_grid, 6),
    )
