"""Sequential multi-provider, multi-key failover for operator-note interpretation.

Tries provider/key candidates in configured priority order. Stops as soon as
one candidate produces an interpretation that is BOTH structurally complete
(right number of entries) AND passes deterministic guardrail validation for
every note without needing a safe no_op downgrade. If no candidate achieves a
fully clean result, the best partial (guardrail-safe) result seen so far is
used instead of failing the whole request -- a guardrail downgrade is already
the sanctioned safe outcome, never a crash and never an invented directive.
Only a total failure (every candidate raised, none produced anything
parseable) becomes a controlled error.
"""

import logging
import time
from dataclasses import dataclass

from app import config
from app.directives.interpreter import interpret_operator_notes
from app.directives.validator import ValidatedInterpretation, validate_interpretations
from app.llm.gemini import GeminiProvider
from app.llm.grok import GrokProvider
from app.llm.provider import (
    AuthenticationFailure,
    LLMProviderError,
    ModelNotFoundError,
    RetryableProviderFailure,
)

logger = logging.getLogger("gridwise.llm.failover")

PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}

# Test-only injection seam: when set, run_interpretation bypasses real provider
# construction entirely and uses this LLMProvider instance directly. Tests set
# this instead of configuring real API keys. Never set in production.
_OVERRIDE_PROVIDER = None


@dataclass
class Candidate:
    provider_name: str
    key_index: int
    api_key: str
    model: str
    fallback_model: str | None


@dataclass
class FailoverResult:
    validated: list[ValidatedInterpretation]
    provider_used: str | None
    attempts: int
    clean: bool


class AllProvidersFailedError(LLMProviderError):
    """Every configured provider/key candidate failed to produce any
    parseable interpretation. No optimizer call is ever made in this case."""


def _provider_keys(name: str) -> list[str]:
    if name == "gemini":
        return config.GEMINI_API_KEYS
    if name == "grok":
        return config.GROK_API_KEYS
    return []


def _provider_models(name: str) -> tuple[str, str | None]:
    if name == "gemini":
        return config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL
    if name == "grok":
        return config.GROK_MODEL, config.GROK_FALLBACK_MODEL
    return "", None


def build_candidates() -> list[Candidate]:
    """Expand LLM_PROVIDER_ORDER x each provider's key pool into an ordered
    candidate list. Providers with no configured key are skipped entirely."""
    candidates: list[Candidate] = []
    for provider_name in config.LLM_PROVIDER_ORDER:
        if provider_name not in PROVIDER_CLASSES:
            continue
        keys = _provider_keys(provider_name)
        model, fallback_model = _provider_models(provider_name)
        for idx, key in enumerate(keys):
            candidates.append(Candidate(provider_name, idx, key, model, fallback_model))
    return candidates


MIN_CALL_SECONDS = 1.0


def _is_clean(raw: list, validated: list[ValidatedInterpretation], expected_count: int) -> bool:
    if not isinstance(raw, list) or len(raw) != expected_count:
        return False
    indices = [e.get("note_index") if isinstance(e, dict) else None for e in raw]
    if sorted(i for i in indices if isinstance(i, int) and not isinstance(i, bool)) != list(
        range(expected_count)
    ):
        return False
    return not any(v.was_downgraded for v in validated)


def _try_candidate(
    candidate: Candidate,
    operator_notes: list[str],
    battery_capacity_kwh: float,
    model: str,
    deadline: float,
) -> tuple[list, list[ValidatedInterpretation]]:
    remaining = deadline - time.monotonic()
    if remaining < MIN_CALL_SECONDS:
        raise RetryableProviderFailure("interpretation time budget exhausted")
    provider_cls = PROVIDER_CLASSES[candidate.provider_name]
    provider = provider_cls(
        api_key=candidate.api_key,
        model=model,
        timeout_seconds=min(config.LLM_TIMEOUT_SECONDS, remaining),
    )
    raw = interpret_operator_notes(provider, operator_notes, battery_capacity_kwh, deadline=deadline)
    validated = validate_interpretations(raw, operator_notes, battery_capacity_kwh)
    return raw, validated


def run_interpretation(operator_notes: list[str], battery_capacity_kwh: float) -> FailoverResult:
    if _OVERRIDE_PROVIDER is not None:
        raw = interpret_operator_notes(_OVERRIDE_PROVIDER, operator_notes, battery_capacity_kwh)
        validated = validate_interpretations(raw, operator_notes, battery_capacity_kwh)
        return FailoverResult(
            validated=validated,
            provider_used="override",
            attempts=1,
            clean=_is_clean(raw, validated, len(operator_notes)),
        )

    candidates = build_candidates()
    if not candidates:
        raise AllProvidersFailedError(
            "No LLM provider is configured (no API key set for any provider in LLM_PROVIDER_ORDER)"
        )

    n = len(operator_notes)
    best: FailoverResult | None = None
    attempts = 0
    max_attempts = max(1, config.LLM_MAX_ATTEMPTS)
    # Hard wall-clock budget for all LLM work so the request stays well inside
    # the judge's 30 s per-request ceiling, however many candidates exist.
    deadline = time.monotonic() + config.LLM_TOTAL_BUDGET_SECONDS

    for candidate in candidates:
        if attempts >= max_attempts or deadline - time.monotonic() < MIN_CALL_SECONDS:
            break

        model = candidate.model
        tried_fallback_model = False
        while True:
            attempts += 1
            try:
                raw, validated = _try_candidate(
                    candidate, operator_notes, battery_capacity_kwh, model, deadline
                )
            except AuthenticationFailure as exc:
                logger.warning(
                    "provider=%s key_index=%d auth failure, moving to next candidate",
                    candidate.provider_name, candidate.key_index,
                )
                break
            except ModelNotFoundError as exc:
                if not tried_fallback_model and candidate.fallback_model and candidate.fallback_model != model:
                    logger.warning(
                        "provider=%s model=%s not found, retrying same key with fallback model=%s",
                        candidate.provider_name, model, candidate.fallback_model,
                    )
                    model = candidate.fallback_model
                    tried_fallback_model = True
                    if attempts >= max_attempts:
                        break
                    continue
                logger.warning(
                    "provider=%s model=%s not found and no usable fallback model, moving on",
                    candidate.provider_name, model,
                )
                break
            except RetryableProviderFailure as exc:
                logger.warning(
                    "provider=%s key_index=%d retryable failure (%s), moving to next candidate",
                    candidate.provider_name, candidate.key_index, type(exc).__name__,
                )
                break
            except LLMProviderError as exc:
                logger.warning(
                    "provider=%s key_index=%d produced no usable output (%s), moving to next candidate",
                    candidate.provider_name, candidate.key_index, type(exc).__name__,
                )
                break

            clean = _is_clean(raw, validated, n)
            result = FailoverResult(
                validated=validated, provider_used=candidate.provider_name, attempts=attempts, clean=clean
            )
            if clean:
                logger.info(
                    "provider=%s key_index=%d model=%s produced a clean interpretation after %d attempt(s)",
                    candidate.provider_name, candidate.key_index, model, attempts,
                )
                return result

            if best is None:
                best = result
            break  # this candidate is exhausted (not clean); move to next candidate

    if best is not None:
        logger.warning(
            "no candidate produced a fully clean interpretation after %d attempt(s); "
            "using best available guardrail-safe result from provider=%s",
            attempts, best.provider_used,
        )
        return best

    raise AllProvidersFailedError(
        f"All {len(candidates)} configured LLM candidate(s) failed after {attempts} attempt(s)"
    )
