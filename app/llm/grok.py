import json
import requests

from app.llm.provider import (
    AuthenticationFailure,
    LLMProvider,
    LLMProviderError,
    ModelNotFoundError,
    RetryableProviderFailure,
)
from app.llm.schema import INTERPRETATION_RESPONSE_SCHEMA

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


class GrokProvider(LLMProvider):
    """xAI Grok adapter. Uses the OpenAI-compatible chat completions API with
    JSON-schema-constrained structured output, translating into the exact
    same canonical shape GeminiProvider produces."""

    name = "grok"

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 20.0):
        if not api_key:
            raise AuthenticationFailure("Grok API key is not configured")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gridwise_interpretations",
                    "schema": INTERPRETATION_RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
        }
        try:
            resp = requests.post(
                GROK_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise RetryableProviderFailure(f"grok request timed out: {exc}") from exc
        except requests.RequestException as exc:
            raise RetryableProviderFailure(f"grok request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise AuthenticationFailure(f"grok returned status {resp.status_code}: unauthorized")
        if resp.status_code == 404:
            raise ModelNotFoundError(f"grok model '{self._model}' not found")
        if resp.status_code == 429:
            raise RetryableProviderFailure("grok rate limit (429) exceeded")
        if resp.status_code >= 500:
            raise RetryableProviderFailure(f"grok server error {resp.status_code}")
        if resp.status_code != 200:
            snippet = resp.text[:300] if resp.text else ""
            raise RetryableProviderFailure(f"grok returned status {resp.status_code}: {snippet}")

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"unexpected grok response shape: {exc}") from exc

        return text
