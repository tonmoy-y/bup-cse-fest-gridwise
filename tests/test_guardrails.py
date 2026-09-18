import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.directives.validator import validate_interpretations


def test_missing_note_defaults_to_no_op():
    notes = ["note a", "note b"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [1, 2], "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert len(result) == 2
    assert result[1].directive_type == "no_op"
    assert result[1].applies is False


def test_invalid_hours_downgrades_to_no_op():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [5, 3], "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert result[0].directive_type == "no_op"


def test_out_of_range_hour_downgrades():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [24], "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert result[0].directive_type == "no_op"


def test_unsupported_directive_type_downgrades():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "shutdown_grid", "hours": [1], "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert result[0].directive_type == "no_op"


def test_solar_reduction_factor_out_of_range_downgrades():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "solar_reduction", "hours": [1], "factor": 1.5, "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert result[0].directive_type == "no_op"


def test_valid_solar_reduction_passes():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": True, "directive_type": "solar_reduction", "hours": [13, 14], "factor": 0.2, "explanation": "x"}]
    result = validate_interpretations(raw, notes)
    assert result[0].directive_type == "solar_reduction"
    assert result[0].structured_adjustment == {"hours": [13, 14], "factor": 0.2}


def test_duplicate_note_index_is_rejected():
    notes = ["note a"]
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [1], "explanation": "first"},
        {"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "second"},
    ]
    result = validate_interpretations(raw, notes)
    # Conflicting mappings for the same note must never reach the optimizer.
    assert result[0].directive_type == "no_op"
    assert result[0].structured_adjustment is None
    assert result[0].was_downgraded is True


def test_no_op_requires_applies_false():
    notes = ["note a"]
    raw = [{"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "irrelevant"}]
    result = validate_interpretations(raw, notes)
    assert result[0].applies is False
    assert result[0].structured_adjustment is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")


def test_reserve_above_capacity_rejected():
    raw = [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
            "hours": [18], "minimum_energy_kwh": 501, "explanation": "x"}]
    result = validate_interpretations(raw, ["keep reserve"], 500.0)
    assert result[0].directive_type == "no_op" and result[0].was_downgraded


def test_reserve_equal_capacity_accepted():
    raw = [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
            "hours": [18], "minimum_energy_kwh": 500, "explanation": "x"}]
    result = validate_interpretations(raw, ["keep reserve"], 500.0)
    assert result[0].structured_adjustment == {"hours": [18], "minimum_energy_kwh": 500.0}


def test_duplicate_index_in_multi_note_output_only_downgrades_that_note():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [1], "explanation": "a"},
        {"note_index": 1, "applies": False, "directive_type": "no_op", "explanation": "b"},
        {"note_index": 1, "applies": True, "directive_type": "no_discharge_window", "hours": [2], "explanation": "c"},
    ]
    result = validate_interpretations(raw, ["n0", "n1"])
    assert result[0].directive_type == "no_charge_window"
    assert result[1].directive_type == "no_op" and result[1].was_downgraded
