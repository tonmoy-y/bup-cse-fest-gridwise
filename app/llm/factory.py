from app import config
from app.llm.provider import LLMProvider
from app.llm.gemini import GeminiProvider

_provider_instance: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    if config.LLM_PROVIDER == "gemini":
        _provider_instance = GeminiProvider(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

    return _provider_instance
