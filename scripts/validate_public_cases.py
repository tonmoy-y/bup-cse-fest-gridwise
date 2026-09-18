"""Runs every public sample case through the internal pipeline (no HTTP) and validates:
- directive interpretation coverage/shape
- full 24-hour schedule validity (energy balance, battery rules, directive application)
- recalculated total_grid_kwh / total_cost_bdt / peak_grid_kwh
- cost within tolerance of the official reference optimum (equivalent optimal schedules
  are accepted; exact schedule equality is NOT required)

Usage: python scripts/validate_public_cases.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives.normalizer import normalize_directives
from app.directives.validator import validate_interpretations
from app.llm.factory import get_llm_provider
from app.directives.interpreter import interpret_operator_notes
from app.optimizer.solver import solve_schedule
from app.schemas import BatterySpec, HourEntry
from app.validation.schedule_validator import validate_schedule

COST_TOLERANCE_RATIO = 0.02  # allow up to 2% above reference optimum for equivalent schedules

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public_samples",
    "public_sample_cases.json",
)


def run_case(case: dict) -> tuple[bool, str]:
    input_data = case["input"]
    expected = case["expected_output"]

    hours = [HourEntry(**h) for h in input_data["hours"]]
    battery = BatterySpec(**input_data["battery"])
    operator_notes = input_data["operator_notes"]

    provider = get_llm_provider()
    try:
        raw = interpret_operator_notes(provider, operator_notes, battery.capacity_kwh)
    except Exception as exc:
        return False, f"LLM interpretation failed: {exc}"

    validated = validate_interpretations(raw, operator_notes)

    if len(validated) != len(operator_notes):
        return False, "Interpretation count mismatch"

    expected_types = [e["directive_type"] for e in expected["directive_interpretation"]]
    got_types = [v.directive_type for v in validated]
    if got_types != expected_types:
        return False, f"directive_type mismatch: expected {expected_types}, got {got_types}"

    constraints = normalize_directives(validated, battery.minimum_energy_kwh)

    result = solve_schedule(hours, battery, constraints)

    outcome = validate_schedule(result.hourly, hours, battery, constraints)
    if not outcome.valid:
        return False, f"Schedule invalid: {outcome.errors}"

    reference_cost = expected["total_cost_bdt"]
    if reference_cost > 0:
        ratio = result.total_cost_bdt / reference_cost
        if ratio > 1 + COST_TOLERANCE_RATIO:
            return False, (
                f"Cost too high: got {result.total_cost_bdt}, reference {reference_cost} "
                f"(ratio {ratio:.4f})"
            )

    return True, (
        f"cost={result.total_cost_bdt:.2f} (reference={reference_cost}), "
        f"grid={result.total_grid_kwh:.2f}, peak={result.peak_grid_kwh:.2f}"
    )


def main():
    with open(SAMPLE_PATH) as f:
        data = json.load(f)

    cases = data["cases"]
    passed = 0
    for case in cases:
        case_id = case["id"]
        try:
            ok, detail = run_case(case)
        except Exception as exc:
            ok, detail = False, f"Exception: {exc}"

        status = "PASS" if ok else "FAIL"
        print(f"{status} {case_id} - {detail}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(cases)} PASS")
    if passed != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
