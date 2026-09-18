import json
import requests

from app.llm.provider import LLMProvider, LLMProviderError

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {
                        "type": "string",
                        "enum": [
                            "solar_reduction",
                            "minimum_battery_reserve",
                            "no_charge_window",
                            "no_discharge_window",
                            "max_grid_window",
                            "no_op",
                        ],
                    },
                    "hours": {"type": "array", "items": {"type": "integer"}},
                    "factor": {"type": "number"},
                    "minimum_energy_kwh": {"type": "number"},
                    "max_grid_kwh": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "applies", "directive_type", "explanation"],
            },
        }
    },
    "required": ["interpretations"],
}


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: float = 20.0):
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not configured")
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
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }
        try:
            resp = requests.post(
                url,
                params={"key": self._api_key},
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        if resp.status_code != 200:
            raise LLMProviderError(f"LLM provider returned status {resp.status_code}")

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"Unexpected LLM response shape: {exc}") from exc

        return text
