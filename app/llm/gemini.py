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

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 20.0):
        if not api_key:
            raise AuthenticationFailure("Gemini API key is not configured")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        url = GEMINI_ENDPOINT_TEMPLATE.format(model=self._model)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "topP": 0,
                "topK": 1,
                "seed": 7,
                "responseMimeType": "application/json",
                "responseSchema": INTERPRETATION_RESPONSE_SCHEMA,
            },
        }
        try:
            resp = requests.post(
                url,
                params={"key": self._api_key},
                json=payload,
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise RetryableProviderFailure(f"gemini request timed out: {exc}") from exc
        except requests.RequestException as exc:
            raise RetryableProviderFailure(f"gemini request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise AuthenticationFailure(f"gemini returned status {resp.status_code}: unauthorized")
        if resp.status_code == 404:
            raise ModelNotFoundError(f"gemini model '{self._model}' not found")
        if resp.status_code == 429:
            raise RetryableProviderFailure("gemini rate limit (429) exceeded")
        if resp.status_code >= 500:
            raise RetryableProviderFailure(f"gemini server error {resp.status_code}")
        if resp.status_code != 200:
            snippet = resp.text[:300] if resp.text else ""
            raise RetryableProviderFailure(
                f"gemini returned status {resp.status_code}: {snippet}"
            )

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"unexpected gemini response shape: {exc}") from exc

        return text
