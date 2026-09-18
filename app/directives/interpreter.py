import json
import re
import time

from app.llm.provider import LLMProvider, LLMProviderError

SYSTEM_PROMPT = """You are the operator-note interpreter for GridWise, a smart-campus energy \
scheduling system. You convert short natural-language operator notes into strict structured \
directives that a deterministic optimizer will apply to a 24-hour energy schedule (hours 0-23).

Supported directive types (use exactly one per note). Each type needs at most ONE extra numeric \
parameter, always placed in the single field called "value":
- solar_reduction: usable solar is reduced during specific hours.
  fields: hours (list of int), value = factor (0..1, the USABLE FRACTION REMAINING). \
"an 80% reduction" -> value = 0.2. "solar at 25% of forecast" -> value = 0.25. "about half of the \
normal/forecast output" -> value = 0.5. "a third of usual output" -> value ≈ 0.33. "reduced by a \
quarter" -> value = 0.75. You MUST always fill in value for this type.
- minimum_battery_reserve: battery energy must stay at or above a level during specific hours.
  fields: hours (list of int), value = minimum_energy_kwh (a kWh amount). If the note states an \
absolute kWh amount, use it directly. If the note states a percentage of battery capacity (e.g. \
"50% of the battery"), compute value = percentage * battery_capacity_kwh using the \
battery_capacity_kwh given below. Never invent a capacity value; only use the one provided.
- no_charge_window: battery charging is disabled during specific hours. fields: hours only \
(no value).
- no_discharge_window: battery discharging is disabled during specific hours. fields: hours only \
(no value).
- max_grid_window: grid import may not exceed a stated kWh amount during specific hours.
  fields: hours (list of int), value = max_grid_kwh (a kWh amount).
- no_op: the note does not affect today's 24-hour energy schedule (distractor / irrelevant note, \
e.g. unrelated campus announcements, deadlines, or events with no energy impact). fields: none.

Rules:
- Interpret every note independently. Return exactly one interpretation per note, in the same \
order as given (note_index starting at 0).
- Time windows are start-inclusive and end-exclusive using whole hours: "1 PM to 3 PM" means \
hours [13, 14]. "6 PM until 9 PM" means hours [18, 19, 20]. Convert 12-hour clock times to the \
24-hour hour index (e.g. 2 AM = 2, 2 PM = 14, noon = 12, midnight = 0).
- hours must be unique integers between 0 and 23 inclusive, in ascending order.
- Never invent demand, solar, tariff, or battery parameters that are not present in the note. \
Never invent a directive type outside the six listed above.
- If a note does not clearly map to one of the five actionable directive types, classify it as \
no_op. When in doubt about relevance to today's energy schedule, prefer no_op.
- For every actionable (non-no_op) directive, applies must be true. For no_op, applies must be \
false and "value" must be absent.
- Leave "value" completely absent for no_charge_window, no_discharge_window, and no_op. Never set \
it to 0 or any placeholder for those types.
- Handle paraphrases, percentages, and equivalent numeric phrasing robustly; do not rely on exact \
wording. Any note describing a physical condition that changes usable solar, battery charge/\
discharge availability, a grid-import limit, or a required battery reserve level for today's \
schedule IS relevant and must NOT be marked no_op, even if phrased indirectly (e.g. maintenance, \
cleaning, outages, inspections, testing, weather, cloud cover all commonly indicate an actionable \
directive). Only notes about unrelated campus/administrative matters (deadlines, events, menus, \
announcements with no physical energy-system effect) are no_op.
- explanation is a short, human-readable one-sentence justification.

Worked examples (for calibration only; do not copy their wording):
- "Rooftop panels will be cleaned from 10 AM to noon, cutting usable solar to about 30%." -> \
solar_reduction, hours [10, 11], value 0.3.
- "No charging is possible between 3 PM and 5 PM due to a hardware fault." -> no_charge_window, \
hours [15, 16], no value.
- "IT will patch the ticketing system this weekend." -> no_op (unrelated to today's energy system).
- "Keep at least 40% of battery capacity available from 8 PM to 10 PM." with battery_capacity_kwh \
= 150 -> minimum_battery_reserve, hours [20, 21], value 60 (0.4 * 150).
- "Cloud cover will leave about half of the forecast solar output from 10 AM until noon." -> \
solar_reduction, hours [10, 11], value 0.5.
- "Grid import must not exceed 180 kWh from 7 PM until 9 PM." -> max_grid_window, hours [19, 20], \
value 180.

Output strictly follows the provided JSON schema: a JSON object with a single key \
"interpretations", an array with exactly one entry per input note in order."""

USER_PROMPT_TEMPLATE = """Battery specification for today (only use this number if a note refers \
to a percentage or fraction of the battery; never use it to invent unrelated directives):
battery_capacity_kwh = {capacity_kwh}

Interpret the following {count} operator note(s) for today's 24-hour energy schedule. Notes \
(0-indexed):
{notes_block}

Return the JSON object now."""


def _build_user_prompt(operator_notes: list[str], battery_capacity_kwh: float) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))
    return USER_PROMPT_TEMPLATE.format(
        capacity_kwh=battery_capacity_kwh, count=len(operator_notes), notes_block=notes_block
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("LLM output is not valid JSON")


def _expand_value_field(entry: dict) -> dict:
    """Map the schema's single generic "value" field to the directive-specific field name
    that the guardrail validator expects, based on the chosen directive_type."""
    if not isinstance(entry, dict) or "value" not in entry:
        return entry
    directive_type = entry.get("directive_type")
    value = entry.pop("value")
    if directive_type == "solar_reduction":
        entry.setdefault("factor", value)
    elif directive_type == "minimum_battery_reserve":
        entry.setdefault("minimum_energy_kwh", value)
    elif directive_type == "max_grid_window":
        entry.setdefault("max_grid_kwh", value)
    return entry


def interpret_operator_notes(
    provider: LLMProvider,
    operator_notes: list[str],
    battery_capacity_kwh: float,
    deadline: float | None = None,
) -> list[dict]:
    """Call the LLM once to interpret all notes; retry once with a correction prompt on failure."""
    user_prompt = _build_user_prompt(operator_notes, battery_capacity_kwh)

    raw_text = provider.generate_json(SYSTEM_PROMPT, user_prompt)
    try:
        parsed = _extract_json(raw_text)
        interpretations = parsed["interpretations"]
        if not isinstance(interpretations, list):
            raise ValueError("interpretations must be a list")
        return [_expand_value_field(e) for e in interpretations]
    except (ValueError, KeyError, TypeError) as first_error:
        if deadline is not None and deadline - time.monotonic() < 2.0:
            raise LLMProviderError(
                f"LLM output invalid and no time left for a correction retry: {first_error}"
            ) from first_error
        correction_prompt = (
            user_prompt
            + "\n\nYour previous response could not be parsed as the required JSON object "
            f"(error: {first_error}). Return ONLY a valid JSON object with the exact "
            '"interpretations" array shape described in the system instructions, with no '
            "extra text."
        )
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining < 2.0:
                raise LLMProviderError("LLM output invalid and no time left for a correction retry")
            if hasattr(provider, "_timeout"):
                provider._timeout = min(provider._timeout, remaining)
        try:
            raw_retry = provider.generate_json(SYSTEM_PROMPT, correction_prompt)
            parsed_retry = _extract_json(raw_retry)
            interpretations = parsed_retry["interpretations"]
            if not isinstance(interpretations, list):
                raise ValueError("interpretations must be a list")
            return [_expand_value_field(e) for e in interpretations]
        except (ValueError, KeyError, TypeError) as second_error:
            raise LLMProviderError(
                f"LLM output invalid after retry: {second_error}"
            ) from second_error
