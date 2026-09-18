import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives.normalizer import normalize_directives
from app.directives.validator import validate_interpretations
from app.optimizer.solver import solve_schedule
from app.schemas import BatterySpec, HourEntry
from app.validation.schedule_validator import validate_schedule

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public_samples",
    "public_sample_cases.json",
)


def load_cases():
    with open(SAMPLE_PATH) as f:
        return json.load(f)["cases"]


def test_all_public_cases_with_ground_truth_directives():
    cases = load_cases()
    assert len(cases) == 10

    for case in cases:
        input_data = case["input"]
        expected = case["expected_output"]
        hours = [HourEntry(**h) for h in input_data["hours"]]
        battery = BatterySpec(**input_data["battery"])

        # Use the ground-truth directive_interpretation directly (bypasses LLM)
        raw = [
            {
                "note_index": d["note_index"],
                "applies": d["applies"],
                "directive_type": d["directive_type"],
                **(d["structured_adjustment"] or {}),
                "explanation": d["explanation"],
            }
            for d in expected["directive_interpretation"]
        ]
        validated = validate_interpretations(raw, input_data["operator_notes"])

        got_types = [v.directive_type for v in validated]
        expected_types = [d["directive_type"] for d in expected["directive_interpretation"]]
        assert got_types == expected_types, f"{case['id']}: directive type mismatch"

        constraints = normalize_directives(validated, battery.minimum_energy_kwh)
        result = solve_schedule(hours, battery, constraints)

        outcome = validate_schedule(result.hourly, hours, battery, constraints)
        assert outcome.valid, f"{case['id']}: {outcome.errors}"

        ref_cost = expected["total_cost_bdt"]
        assert result.total_cost_bdt <= ref_cost * 1.02 + 1.0, (
            f"{case['id']}: cost {result.total_cost_bdt} exceeds reference {ref_cost}"
        )
        print(f"{case['id']}: OK cost={result.total_cost_bdt:.2f} ref={ref_cost}")


if __name__ == "__main__":
    test_all_public_cases_with_ground_truth_directives()
    print("All public sample cases validated against ground-truth directives.")
