import os


def _split_keys(*env_names: str) -> list[str]:
    """Merge a comma-separated *_API_KEYS var and a single *_API_KEY var into
    one ordered, de-duplicated key pool. Never logs or echoes the values."""
    keys: list[str] = []
    for name in env_names:
        raw = os.environ.get(name, "")
        for part in raw.split(","):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)
    return keys


GEMINI_API_KEYS = _split_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")

GROK_API_KEYS = _split_keys("GROK_API_KEYS", "GROK_API_KEY")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4-fast-non-reasoning")
GROK_FALLBACK_MODEL = os.environ.get("GROK_FALLBACK_MODEL", "grok-3-mini")

LLM_PROVIDER_ORDER = [
    p.strip() for p in os.environ.get("LLM_PROVIDER_ORDER", "gemini,grok").split(",") if p.strip()
]
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))
LLM_MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "28"))

NUMERIC_TOLERANCE = 0.01
