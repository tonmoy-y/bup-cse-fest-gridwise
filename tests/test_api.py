import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.llm import factory as llm_factory
from app.llm.provider import LLMProvider


class FakeProvider(LLMProvider):
    """Deterministic stand-in LLM for testing the API without network calls."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        interpretations = []
        for line in user_prompt.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            idx_str, note = line.split(":", 1)
            if not idx_str.strip().isdigit():
                continue
            idx = int(idx_str.strip())
            note = note.strip().lower()
            if "solar" in note and ("reduc" in note or "%" in note or "clean" in note):
                interpretations.append(
                    {
                        "note_index": idx,
                        "applies": True,
                        "directive_type": "solar_reduction",
                        "hours": [12, 13],
                        "factor": 0.25,
                        "explanation": "Solar reduced during maintenance window.",
                    }
                )
            elif "not charge" in note or "charging" in note and "unavailable" in note:
                interpretations.append(
                    {
                        "note_index": idx,
                        "applies": True,
                        "directive_type": "no_charge_window",
                        "hours": [2, 3, 4],
                        "explanation": "No charging during maintenance.",
                    }
                )
            else:
                interpretations.append(
                    {
                        "note_index": idx,
                        "applies": False,
                        "directive_type": "no_op",
                        "explanation": "Irrelevant note.",
                    }
                )
        return json.dumps({"interpretations": interpretations})


def _load_sample(index=0):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "public_samples",
        "public_sample_cases.json",
    )
    with open(path) as f:
        data = json.load(f)
    return data["cases"][index]["input"]


def test_health():
    from app.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_optimize_energy_happy_path(monkeypatch):
    llm_factory._provider_instance = FakeProvider()
    from app.main import app

    client = TestClient(app)
    payload = _load_sample(0)
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scenario_id"] == payload["scenario_id"]
    assert len(body["directive_interpretation"]) == len(payload["operator_notes"])
    assert len(body["hourly_plan"]) == 24
    assert body["total_grid_kwh"] > 0
    llm_factory._provider_instance = None


def test_optimize_energy_malformed_json():
    from app.main import app

    client = TestClient(app)
    resp = client.post("/optimize-energy", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_optimize_energy_missing_field():
    from app.main import app

    client = TestClient(app)
    resp = client.post("/optimize-energy", json={"scenario_id": "X"})
    assert resp.status_code == 400


if __name__ == "__main__":
    test_health()
    print("PASS test_health")
    test_optimize_energy_happy_path(None)
    print("PASS test_optimize_energy_happy_path")
    test_optimize_energy_malformed_json()
    print("PASS test_optimize_energy_malformed_json")
    test_optimize_energy_missing_field()
    print("PASS test_optimize_energy_missing_field")
