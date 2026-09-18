from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised when the LLM provider fails to produce a usable response."""


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
