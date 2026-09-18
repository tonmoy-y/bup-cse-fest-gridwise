"""Adversarial request-schema tests: hostile /optimize-energy inputs must be
rejected cleanly with 400, never crash, never reach the LLM/optimizer."""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public_samples",
    "public_sample_cases.json",
)


def _base_payload():
    with open(SAMPLE_PATH) as f:
        data = json.load(f)
    return copy.deepcopy(data["cases"][0]["input"])


def _post(payload):
    return client.post("/optimize-energy", json=payload)


def _post_raw(body: bytes):
    return client.post(
        "/optimize-energy", content=body, headers={"Content-Type": "application/json"}
    )


def assert_rejected(payload_or_body, raw=False, label=""):
    resp = _post_raw(payload_or_body) if raw else _post(payload_or_body)
    assert resp.status_code in (400, 422), f"{label}: expected 400/422, got {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    assert "error" in body, f"{label}: missing error field"


# ---------------------------------------------------------------------------
# Top-level body shape
# ---------------------------------------------------------------------------

def test_empty_object():
    assert_rejected({}, label="empty object")


def test_null_body():
    assert_rejected(b"null", raw=True, label="null body")


def test_plain_text_body():
    assert_rejected(b"this is not json", raw=True, label="plain text body")


def test_malformed_json():
    assert_rejected(b'{"scenario_id": "X"', raw=True, label="malformed JSON")


def test_json_array_instead_of_object():
    assert_rejected(b"[1,2,3]", raw=True, label="JSON array body")


# ---------------------------------------------------------------------------
# scenario_id
# ---------------------------------------------------------------------------

def test_missing_scenario_id():
    p = _base_payload()
    del p["scenario_id"]
    assert_rejected(p, label="missing scenario_id")


def test_scenario_id_null():
    p = _base_payload()
    p["scenario_id"] = None
    assert_rejected(p, label="scenario_id null")


def test_scenario_id_number():
    p = _base_payload()
    p["scenario_id"] = 123
    assert_rejected(p, label="scenario_id number")


def test_scenario_id_object():
    p = _base_payload()
    p["scenario_id"] = {"a": 1}
    assert_rejected(p, label="scenario_id object")


# ---------------------------------------------------------------------------
# operator_notes
# ---------------------------------------------------------------------------

def test_missing_operator_notes():
    p = _base_payload()
    del p["operator_notes"]
    assert_rejected(p, label="missing operator_notes")


def test_operator_notes_null():
    p = _base_payload()
    p["operator_notes"] = None
    assert_rejected(p, label="operator_notes null")


def test_operator_notes_string_instead_of_array():
    p = _base_payload()
    p["operator_notes"] = "a single string"
    assert_rejected(p, label="operator_notes string")


def test_operator_notes_empty_array():
    p = _base_payload()
    p["operator_notes"] = []
    assert_rejected(p, label="operator_notes empty array")


def test_operator_notes_too_many():
    p = _base_payload()
    p["operator_notes"] = ["a", "b", "c", "d"]
    assert_rejected(p, label="operator_notes 4 items")


def test_operator_notes_empty_string_item():
    p = _base_payload()
    p["operator_notes"] = [""]
    assert_rejected(p, label="operator_notes empty string item")


def test_operator_notes_whitespace_only():
    p = _base_payload()
    p["operator_notes"] = ["   \t  "]
    assert_rejected(p, label="operator_notes whitespace-only")


def test_operator_notes_item_null():
    p = _base_payload()
    p["operator_notes"] = [None]
    assert_rejected(p, label="operator_notes item null")


def test_operator_notes_item_number():
    p = _base_payload()
    p["operator_notes"] = [123]
    assert_rejected(p, label="operator_notes item number")


def test_operator_notes_item_object():
    p = _base_payload()
    p["operator_notes"] = [{"note": "x"}]
    assert_rejected(p, label="operator_notes item object")


def test_operator_notes_item_array():
    p = _base_payload()
    p["operator_notes"] = [["x"]]
    assert_rejected(p, label="operator_notes item array")


# ---------------------------------------------------------------------------
# hours
# ---------------------------------------------------------------------------

def test_missing_hours():
    p = _base_payload()
    del p["hours"]
    assert_rejected(p, label="missing hours")


def test_hours_null():
    p = _base_payload()
    p["hours"] = None
    assert_rejected(p, label="hours null")


def test_hours_non_array():
    p = _base_payload()
    p["hours"] = "not an array"
    assert_rejected(p, label="hours non-array")


def test_hours_23_entries():
    p = _base_payload()
    p["hours"] = p["hours"][:23]
    assert_rejected(p, label="23 hours")


def test_hours_25_entries():
    p = _base_payload()
    extra = dict(p["hours"][0])
    extra["hour"] = 23
    p["hours"] = p["hours"] + [extra]
    assert_rejected(p, label="25 hours (with dup)")


def test_hours_duplicate_hour():
    p = _base_payload()
    p["hours"][1]["hour"] = p["hours"][0]["hour"]
    assert_rejected(p, label="duplicate hour")


def test_hours_missing_one_hour_value():
    p = _base_payload()
    p["hours"][23]["hour"] = 5  # now hour 23 missing, hour 5 duplicated
    assert_rejected(p, label="missing hour 23 / duplicate hour 5")


def test_hours_out_of_range_high():
    p = _base_payload()
    p["hours"][0]["hour"] = 24
    assert_rejected(p, label="hour = 24")


def test_hours_negative():
    p = _base_payload()
    p["hours"][0]["hour"] = -1
    assert_rejected(p, label="negative hour")


def test_hours_decimal():
    p = _base_payload()
    p["hours"][0]["hour"] = 5.5
    assert_rejected(p, label="decimal hour")


def test_hours_string():
    p = _base_payload()
    p["hours"][0]["hour"] = "5"
    assert_rejected(p, label="string hour")


def test_hours_boolean():
    p = _base_payload()
    p["hours"][0]["hour"] = True
    assert_rejected(p, label="boolean hour")


def test_hours_shuffled_still_valid():
    """Shuffled order (still covering 0..23 exactly once) must be ACCEPTED."""
    p = _base_payload()
    import random

    shuffled = p["hours"][:]
    random.Random(42).shuffle(shuffled)
    p["hours"] = shuffled
    resp = _post(p)
    # LLM call will fail (no network in this suite context is fine either way,
    # but it must NOT be rejected at the schema layer with 400).
    assert resp.status_code != 400, f"shuffled valid hours wrongly rejected: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# hour entry fields
# ---------------------------------------------------------------------------

def test_hour_missing_demand():
    p = _base_payload()
    del p["hours"][0]["demand_kwh"]
    assert_rejected(p, label="missing demand_kwh")


def test_hour_missing_solar():
    p = _base_payload()
    del p["hours"][0]["solar_kwh"]
    assert_rejected(p, label="missing solar_kwh")


def test_hour_missing_tariff():
    p = _base_payload()
    del p["hours"][0]["tariff_bdt_per_kwh"]
    assert_rejected(p, label="missing tariff_bdt_per_kwh")


def test_hour_demand_null():
    p = _base_payload()
    p["hours"][0]["demand_kwh"] = None
    assert_rejected(p, label="demand_kwh null")


def test_hour_demand_string():
    p = _base_payload()
    p["hours"][0]["demand_kwh"] = "ninety"
    assert_rejected(p, label="demand_kwh string")


def test_hour_demand_negative():
    p = _base_payload()
    p["hours"][0]["demand_kwh"] = -10
    assert_rejected(p, label="negative demand_kwh")


def test_hour_solar_negative():
    p = _base_payload()
    p["hours"][0]["solar_kwh"] = -1
    assert_rejected(p, label="negative solar_kwh")


def test_hour_tariff_negative():
    p = _base_payload()
    p["hours"][0]["tariff_bdt_per_kwh"] = -5
    assert_rejected(p, label="negative tariff_bdt_per_kwh")


def test_hour_zero_values_accepted():
    p = _base_payload()
    p["hours"][0]["demand_kwh"] = 0
    p["hours"][0]["solar_kwh"] = 0
    p["hours"][0]["tariff_bdt_per_kwh"] = 0
    resp = _post(p)
    assert resp.status_code != 400, f"zero values wrongly rejected: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# battery
# ---------------------------------------------------------------------------

def test_missing_battery():
    p = _base_payload()
    del p["battery"]
    assert_rejected(p, label="missing battery")


def test_battery_null():
    p = _base_payload()
    p["battery"] = None
    assert_rejected(p, label="battery null")


def test_battery_missing_capacity():
    p = _base_payload()
    del p["battery"]["capacity_kwh"]
    assert_rejected(p, label="battery missing capacity_kwh")


def test_battery_capacity_zero():
    p = _base_payload()
    p["battery"]["capacity_kwh"] = 0
    assert_rejected(p, label="battery capacity_kwh = 0")


def test_battery_initial_exceeds_capacity():
    p = _base_payload()
    p["battery"]["initial_energy_kwh"] = p["battery"]["capacity_kwh"] + 100
    assert_rejected(p, label="initial_energy_kwh > capacity_kwh")


def test_battery_minimum_exceeds_capacity():
    p = _base_payload()
    p["battery"]["minimum_energy_kwh"] = p["battery"]["capacity_kwh"] + 100
    assert_rejected(p, label="minimum_energy_kwh > capacity_kwh")


def test_battery_negative_max_charge():
    p = _base_payload()
    p["battery"]["max_charge_kwh_per_hour"] = -5
    assert_rejected(p, label="negative max_charge_kwh_per_hour")


def test_battery_negative_max_discharge():
    p = _base_payload()
    p["battery"]["max_discharge_kwh_per_hour"] = -5
    assert_rejected(p, label="negative max_discharge_kwh_per_hour")


def test_battery_zero_rates_accepted():
    """Zero charge/discharge rate is unusual but structurally valid; must not be rejected."""
    p = _base_payload()
    p["battery"]["max_charge_kwh_per_hour"] = 0
    p["battery"]["max_discharge_kwh_per_hour"] = 0
    p["battery"]["initial_energy_kwh"] = p["battery"]["minimum_energy_kwh"]
    resp = _post(p)
    assert resp.status_code != 400, f"zero battery rates wrongly rejected: {resp.text[:200]}"


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
    print(f"\n{len(failures)} failing tests" if failures else "\nAll schema tests passed")
    if failures:
        sys.exit(1)
