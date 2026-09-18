"""Multi-provider failover tests. Uses fake provider CLASSES injected into
app.llm.failover.PROVIDER_CLASSES (and fake key pools via monkeypatched
config) so the full candidate-expansion / classification / stop-on-success
logic in failover.run_interpretation is exercised end-to-end, exactly as it
runs in production -- only the network call itself is faked."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config
from app.llm import failover
from app.llm.provider import (
    AuthenticationFailure,
    LLMProvider,
    ModelNotFoundError,
    RetryableProviderFailure,
)

VALID_NO_OP = json.dumps(
    {"interpretations": [{"note_index": 0, "applies": False, "directive_type": "no_op", "explanation": "x"}]}
)
VALID_DIRECTIVE = json.dumps(
    {
        "interpretations": [
            {"note_index": 0, "applies": True, "directive_type": "no_charge_window", "hours": [1, 2], "explanation": "x"}
        ]
    }
)


class CallLog:
    """Shared call tracker across fake provider instances within one test."""

    def __init__(self):
        self.calls = []  # list of (provider_name, key)


class ScriptedProvider(LLMProvider):
    """A provider whose behavior is scripted per (provider_name, api_key)."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 20.0):
        self.api_key = api_key
        self.model = model

    def generate_json(self, system_prompt, user_prompt):
        raise NotImplementedError  # overridden per-test via a script table


def _make_provider_class(name: str, script: dict, log: CallLog):
    """script maps api_key -> either a response string, or an Exception instance/class to raise."""

    class _Provider(LLMProvider):
        provider_name = name

        def __init__(self, api_key, model, timeout_seconds=20.0):
            self.api_key = api_key
            self.model = model

        def generate_json(self, system_prompt, user_prompt):
            log.calls.append((self.provider_name, self.api_key, self.model))
            action = script.get(self.api_key)
            if action is None:
                raise RetryableProviderFailure(f"no script entry for key {self.api_key}")
            if isinstance(action, Exception):
                raise action
            if isinstance(action, type) and issubclass(action, Exception):
                raise action(f"scripted failure for {self.api_key}")
            return action

    return _Provider


def _setup(monkeypatch, gemini_script=None, grok_script=None, order=("gemini", "grok"), max_attempts=6):
    log = CallLog()
    monkeypatch.setattr(config, "LLM_PROVIDER_ORDER", list(order))
    monkeypatch.setattr(config, "LLM_MAX_ATTEMPTS", max_attempts)
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODEL", "gemini-test-fallback")
    monkeypatch.setattr(config, "GROK_MODEL", "grok-test")
    monkeypatch.setattr(config, "GROK_FALLBACK_MODEL", "grok-test-fallback")

    if gemini_script is not None:
        monkeypatch.setattr(config, "GEMINI_API_KEYS", list(gemini_script.keys()))
        monkeypatch.setitem(failover.PROVIDER_CLASSES, "gemini", _make_provider_class("gemini", gemini_script, log))
    else:
        monkeypatch.setattr(config, "GEMINI_API_KEYS", [])

    if grok_script is not None:
        monkeypatch.setattr(config, "GROK_API_KEYS", list(grok_script.keys()))
        monkeypatch.setitem(failover.PROVIDER_CLASSES, "grok", _make_provider_class("grok", grok_script, log))
    else:
        monkeypatch.setattr(config, "GROK_API_KEYS", [])

    return log


# A minimal fake "monkeypatch" so this file works standalone (python file.py)
# without pytest, matching the other tests/*.py files in this repo.
class _FakeMonkeypatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setitem(self, d, key, value):
        had = key in d
        old = d.get(key)
        self._undo.append((d, key, old if had else "__DELETE__"))
        d[key] = value

    def undo(self):
        for obj, name, old in reversed(self._undo):
            if isinstance(obj, dict):
                if old == "__DELETE__":
                    obj.pop(name, None)
                else:
                    obj[name] = old
            else:
                setattr(obj, name, old)
        self._undo.clear()


def run_test(fn):
    mp = _FakeMonkeypatch()
    try:
        fn(mp)
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# 1. Gemini success => Grok not called
# ---------------------------------------------------------------------------

def test_gemini_success_grok_not_called(mp):
    log = _setup(mp, gemini_script={"g1": VALID_DIRECTIVE}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["do not charge from 1am to 3am"], 200.0)
    assert result.clean
    assert result.provider_used == "gemini"
    assert log.calls == [("gemini", "g1", "gemini-test")]


# ---------------------------------------------------------------------------
# 2 & 3. Gemini key rotation on auth failure
# ---------------------------------------------------------------------------

def test_gemini_key1_auth_failure_falls_to_key2(mp):
    log = _setup(
        mp,
        gemini_script={"g1": AuthenticationFailure("bad key"), "g2": VALID_DIRECTIVE},
        grok_script={"x1": VALID_DIRECTIVE},
    )
    result = failover.run_interpretation(["do not charge from 1am to 3am"], 200.0)
    assert result.clean
    assert result.provider_used == "gemini"
    assert [c[:2] for c in log.calls] == [("gemini", "g1"), ("gemini", "g2")]  # Grok never reached


# ---------------------------------------------------------------------------
# 4. All Gemini keys fail => Grok called
# ---------------------------------------------------------------------------

def test_all_gemini_keys_fail_grok_called(mp):
    log = _setup(
        mp,
        gemini_script={"g1": RetryableProviderFailure("down"), "g2": RetryableProviderFailure("down")},
        grok_script={"x1": VALID_DIRECTIVE},
    )
    result = failover.run_interpretation(["do not charge from 1am to 3am"], 200.0)
    assert result.clean
    assert result.provider_used == "grok"
    providers_called = [c[0] for c in log.calls]
    assert providers_called == ["gemini", "gemini", "grok"]


# ---------------------------------------------------------------------------
# 5 & 6 & 7. 429 / timeout / 500 all trigger Grok fallback
# ---------------------------------------------------------------------------

def test_gemini_429_falls_back_to_grok(mp):
    _setup(mp, gemini_script={"g1": RetryableProviderFailure("429")}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


def test_gemini_timeout_falls_back_to_grok(mp):
    _setup(mp, gemini_script={"g1": RetryableProviderFailure("timeout")}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


def test_gemini_500_falls_back_to_grok(mp):
    _setup(mp, gemini_script={"g1": RetryableProviderFailure("server error")}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


# ---------------------------------------------------------------------------
# 8 & 9 & 10. Malformed / invalid-directive / guardrail-failure responses reject + fall back
# ---------------------------------------------------------------------------

def test_gemini_malformed_json_falls_back(mp):
    _setup(mp, gemini_script={"g1": "not json at all {{{"}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


def test_gemini_invalid_directive_falls_back():
    invalid = json.dumps(
        {"interpretations": [{"note_index": 0, "applies": True, "directive_type": "shutdown_grid", "hours": [1], "explanation": "x"}]}
    )
    run_test(lambda mp: _check_invalid_falls_back(mp, invalid))


def _check_invalid_falls_back(mp, invalid_response):
    _setup(mp, gemini_script={"g1": invalid_response}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


def test_gemini_valid_json_but_guardrail_failure_falls_back(mp):
    # dup hours [13,13,14] and factor > 1: both guardrail-invalid
    bad = json.dumps(
        {
            "interpretations": [
                {"note_index": 0, "applies": True, "directive_type": "solar_reduction", "hours": [13, 13, 14], "value": 1.7, "explanation": "x"}
            ]
        }
    )
    _setup(mp, gemini_script={"g1": bad}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


# ---------------------------------------------------------------------------
# 11, 12, 13. Grok-specific behavior
# ---------------------------------------------------------------------------

def test_grok_success_reaches_optimizer_path(mp):
    _setup(mp, gemini_script=None, grok_script={"x1": VALID_DIRECTIVE}, order=("grok",))
    result = failover.run_interpretation(["do not charge"], 200.0)
    assert result.clean
    assert result.provider_used == "grok"
    assert result.validated[0].directive_type == "no_charge_window"


def test_grok_401_falls_to_next_grok_key(mp):
    log = _setup(
        mp,
        gemini_script=None,
        grok_script={"x1": AuthenticationFailure("bad"), "x2": VALID_DIRECTIVE},
        order=("grok",),
    )
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean
    assert [c[1] for c in log.calls] == ["x1", "x2"]


def test_grok_timeout_moves_to_next_candidate_or_controlled_failure(mp):
    _setup(mp, gemini_script=None, grok_script={"x1": RetryableProviderFailure("timeout")}, order=("grok",))
    try:
        failover.run_interpretation(["note"], 200.0)
        assert False, "expected AllProvidersFailedError"
    except failover.AllProvidersFailedError:
        pass


# ---------------------------------------------------------------------------
# 14. All providers fail => controlled error, optimizer never invoked
# ---------------------------------------------------------------------------

def test_all_providers_fail_controlled_error(mp):
    _setup(
        mp,
        gemini_script={"g1": RetryableProviderFailure("down")},
        grok_script={"x1": RetryableProviderFailure("down")},
    )
    try:
        failover.run_interpretation(["note"], 200.0)
        assert False, "expected AllProvidersFailedError"
    except failover.AllProvidersFailedError:
        pass


def test_all_providers_fail_via_api_returns_500_no_optimizer_call(mp):
    """End-to-end: verify main.py surfaces this as a controlled 500, and the
    response contains no schedule (optimizer was never reached)."""
    _setup(
        mp,
        gemini_script={"g1": RetryableProviderFailure("down")},
        grok_script={"x1": RetryableProviderFailure("down")},
    )
    import importlib
    from fastapi.testclient import TestClient

    import app.main as main_module
    importlib.reload(main_module)
    client = TestClient(main_module.app)

    with open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public_samples", "public_sample_cases.json")
    ) as f:
        payload = json.load(f)["cases"][0]["input"]

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 500
    body = resp.json()
    assert "hourly_plan" not in body
    assert "total_cost_bdt" not in body


# ---------------------------------------------------------------------------
# 16. Provider returns 200 (parseable) but invalid schema => not success
# ---------------------------------------------------------------------------

def test_valid_json_invalid_schema_not_counted_as_success(mp):
    # note_index out of range for a 1-note request -> guardrail downgrades -> not clean
    bad = json.dumps(
        {"interpretations": [{"note_index": 5, "applies": True, "directive_type": "no_charge_window", "hours": [1], "explanation": "x"}]}
    )
    _setup(mp, gemini_script={"g1": bad}, grok_script={"x1": VALID_DIRECTIVE})
    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean and result.provider_used == "grok"


# ---------------------------------------------------------------------------
# Model-not-found triggers fallback model retry on the SAME key
# ---------------------------------------------------------------------------

def test_model_not_found_retries_same_key_with_fallback_model(mp):
    calls = []

    class _Provider(LLMProvider):
        provider_name = "gemini"

        def __init__(self, api_key, model, timeout_seconds=20.0):
            self.api_key = api_key
            self.model = model

        def generate_json(self, system_prompt, user_prompt):
            calls.append((self.api_key, self.model))
            if self.model == "gemini-test":
                raise ModelNotFoundError("model retired")
            return VALID_DIRECTIVE

    mp.setattr(config, "LLM_PROVIDER_ORDER", ["gemini"])
    mp.setattr(config, "GEMINI_API_KEYS", ["g1"])
    mp.setattr(config, "GEMINI_MODEL", "gemini-test")
    mp.setattr(config, "GEMINI_FALLBACK_MODEL", "gemini-test-fallback")
    mp.setattr(config, "LLM_MAX_ATTEMPTS", 6)
    mp.setitem(failover.PROVIDER_CLASSES, "gemini", _Provider)

    result = failover.run_interpretation(["note"], 200.0)
    assert result.clean
    assert calls == [("g1", "gemini-test"), ("g1", "gemini-test-fallback")]


# ---------------------------------------------------------------------------
# Bounded attempts: never exceeds LLM_MAX_ATTEMPTS
# ---------------------------------------------------------------------------

def test_bounded_total_attempts_never_exceeded(mp):
    log = _setup(
        mp,
        gemini_script={f"g{i}": RetryableProviderFailure("down") for i in range(10)},
        grok_script={f"x{i}": RetryableProviderFailure("down") for i in range(10)},
        max_attempts=3,
    )
    try:
        failover.run_interpretation(["note"], 200.0)
    except failover.AllProvidersFailedError:
        pass
    assert len(log.calls) <= 3, f"expected at most 3 attempts, got {len(log.calls)}"


# ---------------------------------------------------------------------------
# No providers configured at all => controlled error, no crash
# ---------------------------------------------------------------------------

def test_no_providers_configured_controlled_error(mp):
    mp.setattr(config, "LLM_PROVIDER_ORDER", ["gemini", "grok"])
    mp.setattr(config, "GEMINI_API_KEYS", [])
    mp.setattr(config, "GROK_API_KEYS", [])
    try:
        failover.run_interpretation(["note"], 200.0)
        assert False, "expected AllProvidersFailedError"
    except failover.AllProvidersFailedError:
        pass


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            import inspect

            sig = inspect.signature(fn)
            if "mp" in sig.parameters:
                run_test(fn)
            else:
                fn()
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print(f"\n{len(failures)} failing tests" if failures else "\nAll failover tests passed")
    if failures:
        sys.exit(1)
