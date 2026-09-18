"""Regression tests for defects found during the independent audit."""

import copy
import json
import os
import time

from fastapi.testclient import TestClient

import app.main as main_mod
from app import config
from app.llm import failover
from app.llm.provider import LLMProvider, RetryableProviderFailure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = json.load(open(os.path.join(ROOT, "public_samples", "public_sample_cases.json")))["cases"][0]


class _NoOp(LLMProvider):
    def generate_json(self, system_prompt, user_prompt):
        n = user_prompt.count("\n", user_prompt.index("Notes")) - 2
        return json.dumps({"interpretations": [
            {"note_index": i, "applies": False, "directive_type": "no_op", "explanation": "x"}
            for i in range(n)
        ]})


def _client(monkeypatch):
    monkeypatch.setattr(failover, "_OVERRIDE_PROVIDER", _NoOp())
    return TestClient(main_mod.app, raise_server_exceptions=False)


def test_nan_in_request_is_400_not_crash(monkeypatch):
    raw = json.dumps(CASE["input"]).replace('"demand_kwh": 180', '"demand_kwh": NaN', 1)
    assert "NaN" in raw
    r = _client(monkeypatch).post("/optimize-energy", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_infinity_in_request_is_400(monkeypatch):
    raw = json.dumps(CASE["input"]).replace('"tariff_bdt_per_kwh": 7', '"tariff_bdt_per_kwh": Infinity', 1)
    assert "Infinity" in raw
    r = _client(monkeypatch).post("/optimize-energy", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_infeasible_scenario_is_controlled_422(monkeypatch):
    body = copy.deepcopy(CASE["input"])
    body["battery"]["initial_energy_kwh"] = 0  # below base minimum: neutrality makes it infeasible
    r = _client(monkeypatch).post("/optimize-energy", json=body)
    assert r.status_code == 422
    assert "error" in r.json()


def test_zero_capacity_battery_is_accepted(monkeypatch):
    body = copy.deepcopy(CASE["input"])
    body["battery"].update(capacity_kwh=0, initial_energy_kwh=0, minimum_energy_kwh=0)
    r = _client(monkeypatch).post("/optimize-energy", json=body)
    assert r.status_code == 200
    assert all(p["battery_action"] == "idle" for p in r.json()["hourly_plan"])


def test_total_llm_budget_is_bounded(monkeypatch):
    """Many slow candidates must not push the request past the time budget."""
    calls = []

    class Slow(LLMProvider):
        def __init__(self, api_key, model, timeout_seconds=20.0):
            self.t = timeout_seconds

        def generate_json(self, s, u):
            calls.append(self.t)
            time.sleep(min(self.t, 0.4))
            raise RetryableProviderFailure("timeout")

    monkeypatch.setattr(failover, "_OVERRIDE_PROVIDER", None)
    monkeypatch.setattr(config, "LLM_PROVIDER_ORDER", ["gemini", "grok"])
    monkeypatch.setattr(config, "GEMINI_API_KEYS", ["k1", "k2", "k3"])
    monkeypatch.setattr(config, "GROK_API_KEYS", ["k4", "k5"])
    monkeypatch.setattr(config, "LLM_MAX_ATTEMPTS", 50)
    monkeypatch.setattr(config, "LLM_TOTAL_BUDGET_SECONDS", 1.5)
    monkeypatch.setitem(failover.PROVIDER_CLASSES, "gemini", Slow)
    monkeypatch.setitem(failover.PROVIDER_CLASSES, "grok", Slow)
    start = time.monotonic()
    try:
        failover.run_interpretation(["note"], 100.0)
    except Exception:
        pass
    assert time.monotonic() - start < 2.0
    assert all(t <= 1.5 for t in calls)


def test_duplicate_note_index_is_not_clean():
    raw = [
        {"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "a"},
        {"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "b"},
    ]
    from app.directives.validator import validate_interpretations
    v = validate_interpretations(raw, ["x", "y"])
    assert failover._is_clean(raw, v, 2) is False
