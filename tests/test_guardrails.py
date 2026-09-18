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


def test_duplicate_note_index_uses_first():
    notes = ["note a"]
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [1], "explanation": "first"},
        {"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "second"},
    ]
    result = validate_interpretations(raw, notes)
    assert result[0].explanation == "first"


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
