"""Concurrency / repeated-request / state-leakage tests. Uses a fake LLM
provider (deterministic per note text) so this never touches the network."""

import concurrent.futures
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.llm import failover as llm_failover
from app.llm.provider import LLMProvider

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeterministicFakeProvider(LLMProvider):
    """Always classifies notes containing 'charge' as no_charge_window on hours
    [1,2], everything else as no_op. Pure function of input, no shared state."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        interpretations = []
        for line in user_prompt.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            idx_str, note = line.split(":", 1)
            if not idx_str.strip().isdigit():
                continue
            idx = int(idx_str.strip())
            if "charge" in note.lower():
                interpretations.append(
                    {"note_index": idx, "applies": True, "directive_type": "no_charge_window", "hours": [1, 2], "explanation": "x"}
                )
            else:
                interpretations.append(
                    {"note_index": idx, "applies": False, "directive_type": "no_op", "explanation": "x"}
                )
        return json.dumps({"interpretations": interpretations})


def _load_samples():
    with open(os.path.join(REPO_ROOT, "public_samples", "public_sample_cases.json")) as f:
        data = json.load(f)
    return [c["input"] for c in data["cases"]]


def test_same_request_repeated_50_times_is_stable():
    llm_failover._OVERRIDE_PROVIDER = DeterministicFakeProvider()
    try:
        from app.main import app

        client = TestClient(app)
        samples = _load_samples()
        payload = samples[0]

        results = []
        for _ in range(50):
            resp = client.post("/optimize-energy", json=payload)
            assert resp.status_code == 200
            results.append(resp.json())

        first = results[0]
        for r in results[1:]:
            assert r["scenario_id"] == first["scenario_id"]
            assert r["total_cost_bdt"] == first["total_cost_bdt"]
            assert r["total_grid_kwh"] == first["total_grid_kwh"]
            assert r["directive_interpretation"] == first["directive_interpretation"]
    finally:
        llm_failover._OVERRIDE_PROVIDER = None


def test_alternating_scenarios_do_not_leak_state():
    """Interleave two different scenarios and verify each response matches its
    OWN scenario_id and its own directive interpretation, never the other's."""
    llm_failover._OVERRIDE_PROVIDER = DeterministicFakeProvider()
    try:
        from app.main import app

        client = TestClient(app)
        samples = _load_samples()
        a, b = samples[0], samples[1]

        for _ in range(10):
            resp_a = client.post("/optimize-energy", json=a)
            resp_b = client.post("/optimize-energy", json=b)
            assert resp_a.json()["scenario_id"] == a["scenario_id"]
            assert resp_b.json()["scenario_id"] == b["scenario_id"]
    finally:
        llm_failover._OVERRIDE_PROVIDER = None


def test_concurrent_requests_no_cross_contamination():
    """Fire many concurrent requests across different scenarios and verify each
    response is internally consistent with its own request (no race condition
    mixing scenario_id, directives, or schedules between threads)."""
    llm_failover._OVERRIDE_PROVIDER = DeterministicFakeProvider()
    try:
        from app.main import app

        client = TestClient(app)
        samples = _load_samples()

        def call(payload):
            resp = client.post("/optimize-energy", json=payload)
            return payload["scenario_id"], resp

        tasks = samples * 4  # 40 total concurrent calls across 10 distinct scenarios
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(call, p) for p in tasks]
            outcomes = [f.result() for f in futures]

        for expected_id, resp in outcomes:
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["scenario_id"] == expected_id, (
                f"cross-contamination: expected {expected_id}, got {body['scenario_id']}"
            )
            assert len(body["hourly_plan"]) == 24
    finally:
        llm_failover._OVERRIDE_PROVIDER = None


def test_concurrent_requests_no_intermittent_500s():
    llm_failover._OVERRIDE_PROVIDER = DeterministicFakeProvider()
    try:
        from app.main import app

        client = TestClient(app)
        samples = _load_samples()

        def call(payload):
            return client.post("/optimize-energy", json=payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(call, p) for p in samples * 3]
            statuses = [f.result().status_code for f in futures]

        assert all(s == 200 for s in statuses), f"unexpected statuses: {statuses}"
    finally:
        llm_failover._OVERRIDE_PROVIDER = None


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
    print(f"\n{len(failures)} failing tests" if failures else "\nAll concurrency tests passed")
    if failures:
        sys.exit(1)
