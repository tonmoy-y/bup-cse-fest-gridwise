from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Base class for all LLM provider failures. Never includes secrets."""


class RetryableProviderFailure(LLMProviderError):
    """Transient failure (timeout, connection error, 429, 5xx). The same key
    should not be retried immediately; move to the next candidate."""


class AuthenticationFailure(LLMProviderError):
    """Invalid/unauthorized/forbidden key. This key must never be retried
    again within the request; move to the next key or provider."""


class ModelNotFoundError(LLMProviderError):
    """The configured model id is invalid or unavailable for this key. Try
    the provider's configured fallback model before moving to the next
    candidate."""


class LLMProvider(ABC):
    """Abstract interface for a language-capable generative model.

    The rest of the application depends only on this interface, never on a
    specific vendor SDK, so swapping models/providers is a configuration
    change rather than a rewrite of the optimizer, guardrails, or API.
    """

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text output (expected to be JSON) for the given prompts."""
        raise NotImplementedError
