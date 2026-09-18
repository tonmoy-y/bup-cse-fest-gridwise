"""Security audit: no stack traces, no secrets, no crash on hostile input."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _client():
    from app.main import app

    return TestClient(app)


def test_no_secrets_in_tracked_git_files():
    result = subprocess.run(
        ["git", "-C", REPO_ROOT, "ls-files"], capture_output=True, text=True, check=True
    )
    tracked = result.stdout.splitlines()
    assert ".env" not in tracked, ".env must never be committed"

    forbidden_patterns = ["AIza" + "Sy", "BEGIN RSA " + "PRIVATE KEY", "BEGIN " + "PRIVATE KEY", "xai-" + "AAAA"]
    for path in tracked:
        full = os.path.join(REPO_ROOT, path)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, "r", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        for pattern in forbidden_patterns:
            assert pattern not in content, f"potential secret literal '{pattern}' found in {path}"


def test_env_example_has_no_real_key():
    with open(os.path.join(REPO_ROOT, ".env.example")) as f:
        content = f.read()
    assert "GEMINI_API_KEY=" in content
    for line in content.splitlines():
        if line.startswith("GEMINI_API_KEY="):
            assert line.strip() == "GEMINI_API_KEY=", "found a value after GEMINI_API_KEY= in .env.example"


def test_malformed_json_error_has_no_traceback():
    client = _client()
    resp = client.post(
        "/optimize-energy", content=b"{not valid", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
    text = resp.text.lower()
    for marker in ["traceback", "file \"", "line ", ".py\"", "exception in"]:
        assert marker not in text, f"response leaked internal detail: {marker}"


def test_schema_error_has_no_traceback_or_raw_exception_object():
    client = _client()
    resp = client.post("/optimize-energy", json={"scenario_id": 123})
    assert resp.status_code == 400
    # must be valid JSON (this itself proves no unserializable object leaked through)
    body = resp.json()
    assert "error" in body
    text = json.dumps(body).lower()
    assert "traceback" not in text
    assert "valueerror(" not in text  # raw repr of an exception object would look like this


def test_missing_api_key_error_has_no_key_leak():
    """Even in the LLM-failure path, the response must never include the key,
    a stack trace, or internal file paths."""
    # Force provider construction failure without touching real env/network.
    from app.llm import failover as llm_failover
    from app.llm.provider import LLMProvider, LLMProviderError

    class BrokenProvider(LLMProvider):
        def generate_json(self, system_prompt, user_prompt):
            raise LLMProviderError("LLM provider returned status 403: key=" + "AIza" + "Sy" + "FAKE" * 8 + " forbidden")

    llm_failover._OVERRIDE_PROVIDER = BrokenProvider()
    try:
        client = _client()
        with open(os.path.join(REPO_ROOT, "public_samples", "public_sample_cases.json")) as f:
            payload = json.load(f)["cases"][0]["input"]
        resp = client.post("/optimize-energy", json=payload)
        assert resp.status_code == 500
        text = resp.text
        assert "AIza" + "Sy" not in text, "raw API key leaked into error response"
        assert "traceback" not in text.lower()
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
    print(f"\n{len(failures)} failing tests" if failures else "\nAll security tests passed")
    if failures:
        sys.exit(1)
