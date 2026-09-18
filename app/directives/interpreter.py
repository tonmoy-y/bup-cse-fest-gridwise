import json
import re

from app.llm.provider import LLMProvider, LLMProviderError

SYSTEM_PROMPT = """You are the operator-note interpreter for GridWise, a smart-campus energy \
scheduling system. You convert short natural-language operator notes into strict structured \
directives that a deterministic optimizer will apply to a 24-hour energy schedule (hours 0-23).

Supported directive types (use exactly one per note):
- solar_reduction: usable solar is reduced during specific hours.
  fields: hours (list of int), factor (0..1, the USABLE FRACTION REMAINING; an 80% reduction \
means factor = 0.2; a note stating a remaining percentage, e.g. "solar will be at 25%", means \
factor = 0.25 directly).
- minimum_battery_reserve: battery energy must stay at or above a level during specific hours.
  fields: hours (list of int), minimum_energy_kwh (number). If the note gives a percentage of \
capacity, you do not know the capacity number, so extract the percentage-derived meaning is not \
possible here — only extract minimum_energy_kwh when the note gives or clearly implies an \
absolute kWh value. If the note gives a percentage without an absolute number you cannot compute \
locally, still return minimum_energy_kwh as the best absolute number you can derive from context \
given in the note itself; never invent a capacity value that was not stated in the note.
- no_charge_window: battery charging is disabled during specific hours. fields: hours.
- no_discharge_window: battery discharging is disabled during specific hours. fields: hours.
- max_grid_window: grid import must not exceed a stated kWh amount during specific hours. \
fields: hours, max_grid_kwh (number).
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
false and no numeric fields should be set.
- Handle paraphrases, percentages, and equivalent numeric phrasing robustly; do not rely on exact \
wording.
- explanation is a short, human-readable one-sentence justification.

Output strictly follows the provided JSON schema: a JSON object with a single key \
"interpretations", an array with exactly one entry per input note in order."""

USER_PROMPT_TEMPLATE = """Interpret the following {count} operator note(s) for today's 24-hour \
energy schedule. Notes (0-indexed):
{notes_block}

Return the JSON object now."""


def _build_user_prompt(operator_notes: list[str]) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))
    return USER_PROMPT_TEMPLATE.format(count=len(operator_notes), notes_block=notes_block)


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


def interpret_operator_notes(provider: LLMProvider, operator_notes: list[str]) -> list[dict]:
    """Call the LLM once to interpret all notes; retry once with a correction prompt on failure."""
    user_prompt = _build_user_prompt(operator_notes)

    raw_text = provider.generate_json(SYSTEM_PROMPT, user_prompt)
    try:
        parsed = _extract_json(raw_text)
        interpretations = parsed["interpretations"]
        if not isinstance(interpretations, list):
            raise ValueError("interpretations must be a list")
        return interpretations
    except (ValueError, KeyError, TypeError) as first_error:
        correction_prompt = (
            user_prompt
            + "\n\nYour previous response could not be parsed as the required JSON object "
            f"(error: {first_error}). Return ONLY a valid JSON object with the exact "
            '"interpretations" array shape described in the system instructions, with no '
            "extra text."
        )
        try:
            raw_retry = provider.generate_json(SYSTEM_PROMPT, correction_prompt)
            parsed_retry = _extract_json(raw_retry)
            interpretations = parsed_retry["interpretations"]
            if not isinstance(interpretations, list):
                raise ValueError("interpretations must be a list")
            return interpretations
        except (ValueError, KeyError, TypeError) as second_error:
            raise LLMProviderError(
                f"LLM output invalid after retry: {second_error}"
            ) from second_error
