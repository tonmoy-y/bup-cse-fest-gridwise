# GridWise LLM-Assisted Energy Optimization API

An LLM-assisted 24-hour smart-campus energy scheduling service built for the BUP CSE Fest 2026
Hackathon (Online Preliminary — GridWise LLM challenge). It interprets natural-language operator
notes with a real language model, deterministically validates and normalizes the resulting
directives, and computes a mathematically optimal, guardrail-verified 24-hour grid/solar/battery
schedule.

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
 Energy Data +   │                   │
 Operator Notes ─▶   LLM Interpreter │  (Gemini — natural language → raw JSON directives)
                 └─────────┬─────────┘
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

- Provider: Google Gemini (`gemini-2.5-flash-lite` by default), called once per request with all
  operator notes batched into a single prompt (minimizes latency/cost — never one call per note).
- The model receives a compact system prompt containing the six directive definitions, the
  whole-hour time-window convention, and the `solar_reduction` factor convention. It does **not**
  receive the 24-hour demand/solar/tariff arrays — it only needs to understand the notes.
- Structured output is requested via Gemini's `responseSchema` / `responseMimeType: application/json`
  feature. If parsing fails, one compact correction retry is attempted. If that also fails, guardrails
  safely fall back to `no_op` for the unparseable note(s) rather than crashing or inventing values.
- The LLM is fully isolated behind a small `LLMProvider` interface
  (`app/llm/provider.py` → `app/llm/gemini.py`), so swapping to another provider/model is a
  configuration change, not a rewrite (`app/llm/factory.py` selects the provider from
  `LLM_PROVIDER`).

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
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Environment variables

| Variable          | Required | Default                  | Purpose                          |
|--------------------|----------|---------------------------|-----------------------------------|
| `GEMINI_API_KEY`   | Yes      | —                          | Gemini API key for interpretation |
| `GEMINI_MODEL`     | No       | `gemini-2.5-flash-lite`    | Gemini model id                   |
| `LLM_PROVIDER`     | No       | `gemini`                   | Provider selector (extensible)    |

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` (never commit real keys).

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

## 15. Vercel deployment

This repo is structured for Vercel's Python runtime: `api/index.py` exports the FastAPI `app`
object directly (ASGI), and `vercel.json` routes all traffic to it.

```bash
npm i -g vercel   # if not already installed
vercel login
vercel link
vercel env add GEMINI_API_KEY
vercel env add GEMINI_MODEL   # optional
vercel --prod
```

After deployment, verify:

```bash
curl https://<your-deployment>.vercel.app/health
```

No local filesystem persistence, background workers, or databases are used, so cold starts are
fast and the deployment is stateless.

## 16. Docker build/run (fallback)

```bash
docker build -t gridwise-llm .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here gridwise-llm
curl http://localhost:8000/health
```

The image binds to `0.0.0.0:8000`, takes all configuration via environment variables, and
contains no baked-in secrets.

## 17. Example response

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

## 18. Dependencies

- `fastapi`, `uvicorn` — HTTP API and ASGI server
- `pydantic` — request/response schema validation
- `scipy` — `linprog` (HiGHS) LP solver for the optimizer
- `numpy` — LP matrix construction
- `requests` — Gemini REST API calls

## 19. Known limitations

- The optimizer models the battery as lossless (no round-trip efficiency loss), matching the
  energy-balance equation given in the Problem Statement (§09) — this is not an approximation
  relative to the spec, but would need extension for a more realistic non-ideal battery.
- The LLM call is a runtime dependency: if the configured provider is unreachable, the service
  returns a controlled `500` (per Section 08 "safe failure") rather than an approximate answer.
  A local/backup model can be substituted by implementing the `LLMProvider` interface.
- `minimum_battery_reserve` given as a percentage of capacity is resolved by the LLM using the
  wording in the note itself (the LLM is never shown the battery spec numbers, per the
  low-token-usage requirement); percentage phrasing referring to an unstated capacity number
  cannot be resolved to an absolute kWh value and will be treated conservatively.

## 20. Security notes

- No API keys, tokens, or secrets are committed to this repository (see `.gitignore`).
- `.env.example` documents required variable names only, with no real values.
- The API never returns raw stack traces or internal exception details — all error responses are
  controlled, generic JSON error objects.
- The Docker image takes all secrets via environment variables at runtime; none are baked into
  the image layers.

## 21. Attribution / credits

- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [SciPy](https://scipy.org/)
  (HiGHS LP solver), [NumPy](https://numpy.org/), [Requests](https://requests.readthedocs.io/) —
  open-source libraries used as-is under their respective licenses.
- [Google Gemini API](https://ai.google.dev/) — hosted language model used for operator-note
  interpretation.
- Core architecture, guardrails, optimizer formulation, and validators are original work for this
  submission.
