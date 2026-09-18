# GridWise LLM-Assisted Energy Optimization API

An LLM-assisted 24-hour smart-campus energy scheduling service built for the BUP CSE Fest 2026
Hackathon (Online Preliminary — GridWise LLM challenge). It interprets natural-language operator
notes with a real language model, deterministically validates and normalizes the resulting
directives, and computes a mathematically optimal, guardrail-verified 24-hour grid/solar/battery
schedule.

**Live Deployment URL:** [https://gridwise.tonmoyy.dev/](https://gridwise.tonmoyy.dev/)

## 1. Overview

```
GET  /health
POST /optimize-energy
```

Given a 24-hour demand/solar/tariff scenario, a battery specification, and 1–3 natural-language
operator notes, the service:

1. Interprets every operator note with a Gemini language model (mandatory LLM step).
2. Converts relevant notes into exactly one of five supported structured directives (or `no_op`).
3. Deterministically validates the LLM output (guardrails) — invalid or unsupported output is
   safely downgraded to `no_op`, never silently invented or allowed to crash the service.
4. Normalizes valid directives into per-hour optimizer constraints.
5. Solves a linear program (`scipy.optimize.linprog`, HiGHS) that minimizes total grid electricity
   cost over 24 hours subject to demand, solar, battery, and directive constraints.
6. Independently replays and validates the resulting schedule against every GridWise energy rule
   before it is ever returned.

## 2. Architecture

```
                 ┌───────────────────┐
 Energy Data +   │  LLM Interpreter  │  Gemini (primary) → Grok (fallback)
 Operator Notes ─▶  + Failover       │  natural language → raw JSON directives,
                 └─────────┬─────────┘  first candidate to pass guardrails wins
                           ▼
                 ┌───────────────────┐
                 │ Guardrail Validator│  (deterministic — schema, hours, ranges, applies-rules)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │Directive Normalizer│  (deterministic — per-hour solar/reserve/charge/grid limits)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │   Math Optimizer   │  (deterministic — linprog / HiGHS LP, cost-minimizing)
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │  Final Validator   │  (deterministic — independent replay of every hour)
                 └─────────┬─────────┘
                           ▼
                    API Response (JSON)
```

The LLM only interprets language. It never touches the optimization math, never sees demand/solar/
tariff numbers, and never receives public sample answers baked into its prompt. Every downstream
step is deterministic Python and independently checkable.

## 3. How the LLM is used

- Primary provider: Google Gemini (`gemini-3.5-flash-lite` by default), called once per request
  with all operator notes batched into a single prompt (minimizes latency/cost — never one call
  per note). Optional secondary provider: xAI Grok, used only on Gemini failure/exhaustion.
- The model receives a compact system prompt containing the six directive definitions, the
  whole-hour time-window convention, and the `solar_reduction` factor convention, plus the
  battery's `capacity_kwh` (needed to resolve percentage-based reserve notes). It does **not**
  receive the 24-hour demand/solar/tariff arrays — it only needs to understand the notes.
- Structured output is requested via each provider's JSON-schema-constrained mode (Gemini's
  `responseSchema`, Grok's `response_format: json_schema`), sharing one canonical schema
  (`app/llm/schema.py`) so both providers are held to the exact same shape. If parsing fails, one
  compact correction retry is attempted. If that also fails, guardrails safely fall back to `no_op`
  for the unparseable note(s) rather than crashing or inventing values.
- **Multi-provider failover** (`app/llm/failover.py`): for each request, provider/key candidates
  are tried in `LLM_PROVIDER_ORDER` (default `gemini,grok`), expanding each provider's key pool
  (`GEMINI_API_KEYS` / `GROK_API_KEYS`, or the single `_API_KEY` form). A candidate only counts as
  successful once its output passes the **same deterministic guardrails** described below with no
  note needing a safe-fallback downgrade; the first such candidate wins and no further provider is
  called. Failures are classified so the right thing happens automatically:
  - **Auth failure** (401/403) → that key is never retried; move to the next key/provider.
  - **Rate limit / timeout / 5xx** → move to the next candidate immediately (no same-key retry).
  - **Model not found** (404) → retry the *same key* once with that provider's configured
    `*_FALLBACK_MODEL`, then move on if still unavailable.
  - **Malformed/invalid output** → guardrail-rejected, move to the next candidate.
  Total attempts per request are capped by `LLM_MAX_ATTEMPTS` (default 4) — never unbounded, and
  a normal successful request only ever makes **one** LLM call. If every candidate fails, the
  service returns a controlled `500` (see §16) rather than fabricating a directive.
- The LLM is fully isolated behind a small `LLMProvider` interface
  (`app/llm/provider.py` → `app/llm/gemini.py` / `app/llm/grok.py`); the interpreter, guardrails,
  optimizer, and API layer depend only on that interface and never on a vendor SDK, so adding a
  third provider is a new ~70-line adapter file plus a config entry, not a rewrite.

## 4. Supported directives

| directive_type            | Meaning                                             | structured_adjustment                          |
|----------------------------|------------------------------------------------------|-------------------------------------------------|
| `solar_reduction`          | Usable solar reduced during specific hours            | `{"hours": [...], "factor": number}`             |
| `minimum_battery_reserve`  | Battery must stay at/above a level during specific hours | `{"hours": [...], "minimum_energy_kwh": number}` |
| `no_charge_window`         | Battery charging disabled during specific hours       | `{"hours": [...]}`                               |
| `no_discharge_window`      | Battery discharging disabled during specific hours    | `{"hours": [...]}`                               |
| `max_grid_window`          | Grid import capped during specific hours              | `{"hours": [...], "max_grid_kwh": number}`       |
| `no_op`                    | Note has no effect on today's schedule                | `null`                                           |

Time windows are start-inclusive/end-exclusive whole hours (e.g. "1 PM to 3 PM" → `[13, 14]`).
`solar_reduction.factor` is the **usable fraction remaining** (an 80% reduction → `factor = 0.2`).

## 5. Optimization approach

The schedule is computed as a **linear program**, not a greedy heuristic — greedy battery
dispatch is not provably optimal once `minimum_battery_reserve`, `max_grid_window`, and
charge/discharge windows can all bind simultaneously. Per hour `h`, five continuous variables are
solved jointly across all 24 hours: `grid[h]`, `solar_used[h]`, `charge[h]`, `discharge[h]`,
`battery_energy_after[h]`, subject to:

- Energy balance: `grid + solar_used + discharge = demand + charge`
- Battery recursion: `energy[h] = energy[h-1] + charge[h] - discharge[h]`
- Battery bounds: `active_reserve[h] <= energy[h] <= capacity`
- Rate limits: `charge[h] <= max_charge_kwh_per_hour`, `discharge[h] <= max_discharge_kwh_per_hour`
  (forced to 0 inside `no_charge_window` / `no_discharge_window` hours)
- Solar limit: `0 <= solar_used[h] <= effective_solar[h]`
- Grid cap: `grid[h] <= max_grid_kwh` inside `max_grid_window` hours
- End-of-day neutrality: `energy[23] = initial_energy_kwh`

Objective: `minimize sum(tariff[h] * grid[h])`, solved with `scipy.optimize.linprog(method="highs")`.
This is exact for the stated linear model — no heuristic approximation. Correctness (feasibility)
is established by the final validator before cost is ever reported.

## 6. Guardrail strategy

`app/directives/validator.py` never trusts raw LLM output. Per Problem Statement §08, every entry
is checked for: allowed `directive_type`, correct `applies` semantics, valid unique-ascending
`hours` in `[0,23]`, required numeric fields with valid ranges (e.g. `factor ∈ [0,1]`), and correct
note-to-index mapping with no duplicates/omissions. Any entry that fails any check is safely
downgraded to `applies=false, directive_type="no_op"` for that specific note — the service never
invents a directive and never crashes on bad model output.

## 7. Final validation strategy

`app/validation/schedule_validator.py` independently replays the optimizer's own output hour by
hour — energy balance, effective-solar limits, battery bounds/rates, active reserve, no-charge/
no-discharge/max-grid directive enforcement, and end-of-day neutrality — using only the validated
directives and the original request data (not the optimizer's internal state). If this fails, the
API returns a controlled `500` rather than an incorrect schedule.

## 8. Local setup

```bash
git clone https://github.com/tonmoy-y/bup-cse-fest-gridwise.git
cd bup-cse-fest-gridwise
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Environment variables

| Variable                 | Required | Default                       | Purpose                                          |
|---------------------------|----------|---------------------------------|----------------------------------------------------|
| `LLM_PROVIDER_ORDER`      | No       | `gemini,grok`                    | Failover priority order (providers with no key configured are skipped) |
| `GEMINI_API_KEY`          | Yes*     | —                                | Single Gemini key                                  |
| `GEMINI_API_KEYS`         | No       | —                                | Comma-separated Gemini key pool (merged with the above) |
| `GEMINI_MODEL`            | No       | `gemini-3.5-flash-lite`          | Gemini model id                                    |
| `GEMINI_FALLBACK_MODEL`   | No       | `gemini-2.0-flash`               | Used if the primary Gemini model id 404s            |
| `GROK_API_KEY`            | No       | —                                | Single Grok (xAI) key                              |
| `GROK_API_KEYS`           | No       | —                                | Comma-separated Grok key pool                      |
| `GROK_MODEL`              | No       | `grok-4-fast-non-reasoning`      | Grok model id                                      |
| `GROK_FALLBACK_MODEL`     | No       | `grok-3-mini`                    | Used if the primary Grok model id 404s              |
| `LLM_TIMEOUT_SECONDS`     | No       | `20`                             | Per-call HTTP timeout                              |
| `LLM_MAX_ATTEMPTS`        | No       | `4`                              | Hard cap on total LLM attempts per request          |

*At least one provider needs at least one key configured; Gemini-only is a fully valid setup for
local development, Grok is purely an optional resilience layer.

Copy `.env.example` to `.env` and fill in your key(s) (never commit real keys).

## 10. Gemini API setup

1. Create a key at https://aistudio.google.com/apikey.
2. `export GEMINI_API_KEY=your_key_here` (or put it in `.env` and load it with your process
   manager / `export $(cat .env | xargs)`).

## 11. Local run command

```bash
export GEMINI_API_KEY=your_key_here
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 12. `/health` example

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## 13. `/optimize-energy` example

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @public_samples/one_case.json
```

(Extract a single `case.input` object from `public_samples/public_sample_cases.json` into
`public_samples/one_case.json`, or POST any object matching the schema in Section 07 of the
Problem Statement.)

## 14. Public sample validation

```bash
export GEMINI_API_KEY=your_key_here
python scripts/validate_public_cases.py
```

This runs all 10 official public cases through the real internal pipeline (LLM → guardrails →
optimizer → final validator), recalculates totals independently, and prints `PASS`/`FAIL` per
case plus a final `N/10 PASS` summary. It does **not** hard-code any public case's expected
schedule — it only checks structural correctness and that recalculated cost is within tolerance
of the published reference optimum (equivalent optimal schedules are accepted, not byte-for-byte
matches).

A pure-math regression test that bypasses the LLM (feeds the ground-truth directives straight into
the optimizer + validator) is also included and reproduces the exact reference cost on all 10
public cases:

```bash
python tests/test_optimizer.py
python tests/test_guardrails.py
python tests/test_api.py   # uses a fake in-process LLM provider, no network/API key needed
```

## 15. Docker build/run (fallback)

```bash
docker build -t gridwise-llm .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here gridwise-llm
curl http://localhost:8000/health
```

The image binds to `0.0.0.0:8000`, takes all configuration via environment variables, and
contains no baked-in secrets.

## 16. Example response

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Solar availability is reduced to 25% during the panel-cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90, "solar_used_kwh": 0, "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": 110}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365,
  "peak_grid_kwh": 175,
  "plan_summary": "Applied 1 operator directive(s) (solar_reduction) and minimized grid cost."
}
```

## 17. Dependencies

- `fastapi`, `uvicorn` — HTTP API and ASGI server
- `pydantic` — request/response schema validation
- `scipy` — `linprog` (HiGHS) LP solver for the optimizer
- `numpy` — LP matrix construction
- `requests` — Gemini and Grok REST API calls

## 18. Known limitations

- The optimizer models the battery as lossless (no round-trip efficiency loss), matching the
  energy-balance equation given in the Problem Statement (§09) — this is not an approximation
  relative to the spec, but would need extension for a more realistic non-ideal battery.
- The LLM call is a runtime dependency: if every configured provider/key candidate is unreachable
  or exhausted, the service returns a controlled `500` (per Section 08 "safe failure") rather than
  an approximate answer. A third provider can be added by implementing the `LLMProvider` interface
  and registering it in `app/llm/failover.py`'s `PROVIDER_CLASSES`.
- Gemini's per-project rate limits are not bypassed by rotating Gemini keys alone (Google enforces
  quota mostly per-project); the Grok fallback exists specifically to survive a Gemini-side outage
  or quota exhaustion, not to multiply Gemini throughput.
- The Grok adapter's strict JSON-schema mode was implemented against xAI's documented
  OpenAI-compatible API shape but has not been exercised against a live Grok account/key in this
  environment; the guardrail layer downstream would still safely reject any malformed Grok output
  either way.
- `minimum_battery_reserve` given as a percentage of capacity is resolved by the LLM using the
  battery's `capacity_kwh` passed alongside the notes (never the full 24-hour scenario, per the
  low-token-usage requirement).

## 19. Security notes

- No API keys, tokens, or secrets are committed to this repository (see `.gitignore`).
- `.env.example` documents required variable names only, with no real values.
- The API never returns raw stack traces or internal exception details — all error responses are
  controlled, generic JSON error objects.
- The Docker image takes all secrets via environment variables at runtime; none are baked into
  the image layers.

## 20. Attribution / credits

- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [SciPy](https://scipy.org/)
  (HiGHS LP solver), [NumPy](https://numpy.org/), [Requests](https://requests.readthedocs.io/) —
  open-source libraries used as-is under their respective licenses.
- [Google Gemini API](https://ai.google.dev/) — primary hosted language model used for
  operator-note interpretation.
- [xAI Grok API](https://x.ai/) — optional secondary/fallback language model, used only if Gemini
  is unconfigured or every Gemini candidate fails for a given request.
- Core architecture, guardrails, optimizer formulation, and validators are original work for this
  submission.
