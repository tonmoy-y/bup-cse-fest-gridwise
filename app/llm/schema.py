"""Canonical structured-output schema shared by every provider adapter.

Both Gemini and Grok are instructed with the same semantic rules (see
app/directives/interpreter.py's SYSTEM_PROMPT) and the same response shape,
so a provider swap never changes what "a valid interpretation" means. Each
adapter may translate this into its own SDK-specific wrapper syntax, but the
object shape itself must stay identical across providers.
"""

INTERPRETATION_RESPONSE_SCHEMA = {
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
                    "value": {
                        "type": "number",
                        "description": (
                            "The single numeric parameter for the chosen directive_type: "
                            "the usable-fraction-remaining for solar_reduction, the kWh reserve "
                            "level for minimum_battery_reserve, or the kWh grid cap for "
                            "max_grid_window. Absent for no_charge_window, no_discharge_window, "
                            "and no_op."
                        ),
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "applies", "directive_type", "explanation"],
            },
        }
    },
    "required": ["interpretations"],
}
